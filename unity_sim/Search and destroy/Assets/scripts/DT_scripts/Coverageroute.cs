using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Reczna trasa coverage: polilinia zlozona z dzieci tego obiektu.
///
/// Trasa jest teraz DEFINICJA ZADANIA - etykieta DT to "na tej pozycji trasa
/// prowadzi tedy", wiec model uczy sie dokladnie tej trasy. Dlatego warto ja
/// zwalidowac przed zbieraniem danych: przejezdnosc kazdego odcinka, katy
/// zakretow wzgledem promienia skretu auta i faktyczne pokrycie pietra.
///
/// Uzycie: prawy przycisk na komponencie -> "Waliduj trase".
/// </summary>
[ExecuteAlways]
public class CoverageRoute : MonoBehaviour
{
    [Header("Ksztalt")]
    [Tooltip("Trasa zamknieta w petle - auto moze jezdzic w kolko bez konca i nie ma problemu 'konca trasy'.")]
    public bool closedLoop = true;

    [Header("Walidacja")]
    [Tooltip("Ile metrow od trasy uznajemy za pokryte. Mniej wiecej szerokosc korytarza / 2.")]
    public float coverageRadius = 1.5f;
    [Tooltip("Rozmiar komorki przy liczeniu pokrycia. Rowny GRID_CELL_SIZE z build_dt_dataset.py.")]
    public float gridCellSize = 1.0f;
    [Tooltip("Kat zakretu powyzej ktorego ostrzegamy. Auto nie skreca w miejscu - ostry zakret oznacza szarpanie albo zaklinowanie.")]
    public float maxTurnAngle = 100f;
    [Tooltip("Jesli sciezka NavMesh miedzy waypointami jest dluzsza niz tyle razy odleglosc w linii prostej, odcinek przechodzi przez sciane.")]
    public float maxDetourRatio = 1.4f;

    [Header("Podglad")]
    public bool showGizmos = true;
    public bool showCoverage = false;
    [Tooltip("Numery waypointow przy gizmach - ulatwia znalezienie punktu po indeksie z walidacji. Co ile punktow pokazac etykiete.")]
    public int labelEvery = 5;

    [Header("Podglad w Game View (LineRenderer)")]
    [Tooltip("Gizma sa widoczne tylko w Scene View. LineRenderer rysuje trase takze w Game View i w trakcie jazdy.")]
    public bool useLineRenderer = false;
    public float lineWidth = 0.08f;
    public Color lineColor = new Color(0f, 1f, 1f, 0.8f);
    public float lineHeightOffset = 0.05f;

    private LineRenderer preview;

    // Cache polilinii. Bez niego kazde wywolanie GetPolyline() alokowalo nowa
    // liste, a AutoExplorer wola PointAtDistance ~17 razy na klatke - czyli
    // kilkanascie alokacji 64-elementowych list co klatke, prosto do GC.
    private readonly List<Vector3> cachedPts = new List<Vector3>();
    private readonly List<float> cachedCum = new List<float>();
    private float cachedLength = 0f;
    private int cacheFrame = -1;

    private void EnsureCache()
    {
        if (cacheFrame == Time.frameCount && cachedPts.Count > 0) return;
        cacheFrame = Time.frameCount;

        cachedPts.Clear();
        cachedCum.Clear();
        foreach (Transform child in transform)
        {
            if (child.name == "__routePreview" || !child.gameObject.activeSelf) continue;
            cachedPts.Add(child.position);
        }
        if (closedLoop && cachedPts.Count >= 2) cachedPts.Add(cachedPts[0]);

        float acc = 0f;
        for (int i = 0; i < cachedPts.Count; i++)
        {
            if (i > 0) acc += Flat(cachedPts[i] - cachedPts[i - 1]).magnitude;
            cachedCum.Add(acc);
        }
        cachedLength = acc;
    }

    /// <summary>Wymusza odswiezenie cache po recznej edycji punktow.</summary>
    public void InvalidateCache() => cacheFrame = -1;

    private readonly List<int> badSegments = new List<int>();
    private readonly List<int> sharpTurns = new List<int>();
    private readonly List<Vector3> uncoveredCells = new List<Vector3>();

    void OnEnable()
    {
        if (useLineRenderer) BuildPreview();
    }

    void Update()
    {
        // trasa moze byc edytowana w trakcie - odswiezamy podglad
        if (useLineRenderer && preview != null && preview.positionCount != GetPolyline().Count)
            BuildPreview();
    }

    [ContextMenu("Zbuduj podglad trasy (LineRenderer)")]
    public void BuildPreview()
    {
        var existing = transform.Find("__routePreview");
        if (existing != null)
        {
            preview = existing.GetComponent<LineRenderer>();
        }
        else
        {
            var go = new GameObject("__routePreview");
            go.hideFlags = HideFlags.DontSave;
            go.transform.SetParent(transform, false);
            preview = go.AddComponent<LineRenderer>();
            preview.material = new Material(Shader.Find("Sprites/Default"));
        }

        var p = GetPolyline();
        preview.useWorldSpace = true;
        preview.widthMultiplier = lineWidth;
        preview.startColor = preview.endColor = lineColor;
        preview.positionCount = p.Count;
        for (int i = 0; i < p.Count; i++)
            preview.SetPosition(i, p[i] + Vector3.up * lineHeightOffset);
    }

    [ContextMenu("Usun podglad trasy")]
    public void ClearPreview()
    {
        var existing = transform.Find("__routePreview");
        if (existing != null) DestroyImmediate(existing.gameObject);
        preview = null;
    }

    // =======================================================================
    // API dla AutoExplorera
    // =======================================================================

    /// <summary>Zwraca WEWNETRZNY bufor - nie modyfikuj go z zewnatrz.</summary>
    public List<Vector3> GetPolyline()
    {
        EnsureCache();
        return cachedPts;
    }

    public float TotalLength()
    {
        EnsureCache();
        return cachedLength;
    }

    /// <summary>Punkt na trasie w odleglosci d od poczatku, mierzonej po luku.
    /// Przy closedLoop d zawija sie modulo dlugosc trasy.</summary>
    public Vector3 PointAtDistance(float d)
    {
        EnsureCache();
        if (cachedPts.Count < 2) return transform.position;

        if (closedLoop && cachedLength > 1e-3f) d = Mathf.Repeat(d, cachedLength);
        d = Mathf.Clamp(d, 0f, cachedLength);

        // binarne wyszukiwanie po skumulowanych dlugosciach zamiast liniowego
        int lo = 0, hi = cachedCum.Count - 1;
        while (lo < hi - 1)
        {
            int mid = (lo + hi) / 2;
            if (cachedCum[mid] <= d) lo = mid; else hi = mid;
        }
        float seg = cachedCum[hi] - cachedCum[lo];
        return Vector3.Lerp(cachedPts[lo], cachedPts[hi],
                            seg > 1e-4f ? (d - cachedCum[lo]) / seg : 0f);
    }

    // =======================================================================
    // Walidacja
    // =======================================================================

    [ContextMenu("Waliduj trase")]
    public void Validate()
    {
        InvalidateCache();
        badSegments.Clear();
        sharpTurns.Clear();

        var p = GetPolyline();
        if (p.Count < 3)
        {
            Debug.LogWarning("[CoverageRoute] Za malo waypointow (min. 3).");
            return;
        }

        // --- przejezdnosc odcinkow ---
        var path = new NavMeshPath();
        for (int i = 0; i < p.Count - 1; i++)
        {
            bool ok = NavMesh.SamplePosition(p[i], out NavMeshHit a, 2f, NavMesh.AllAreas)
                   && NavMesh.SamplePosition(p[i + 1], out NavMeshHit b, 2f, NavMesh.AllAreas)
                   && NavMesh.CalculatePath(a.position, b.position, NavMesh.AllAreas, path)
                   && path.status == NavMeshPathStatus.PathComplete;

            if (ok)
            {
                float navLen = 0f;
                for (int k = 0; k < path.corners.Length - 1; k++)
                    navLen += Flat(path.corners[k + 1] - path.corners[k]).magnitude;
                float straight = Flat(p[i + 1] - p[i]).magnitude;
                if (straight > 0.1f && navLen / straight > maxDetourRatio) ok = false;
            }
            if (!ok) badSegments.Add(i);
        }

        // --- katy zakretow ---
        int last = closedLoop ? p.Count - 1 : p.Count - 1;
        for (int i = 1; i < last; i++)
        {
            Vector3 inDir = Flat(p[i] - p[i - 1]).normalized;
            Vector3 outDir = Flat(p[i + 1] - p[i]).normalized;
            if (inDir.sqrMagnitude < 0.1f || outDir.sqrMagnitude < 0.1f) continue;
            if (Vector3.Angle(inDir, outDir) > maxTurnAngle) sharpTurns.Add(i);
        }

        // --- pokrycie pietra ---
        float covered = ComputeCoverage(out int total, out int hit);

        Debug.Log($"[CoverageRoute] Waypointow: {p.Count - (closedLoop ? 1 : 0)}, "
                + $"dlugosc {TotalLength():F1} m, petla={closedLoop}");
        Debug.Log($"[CoverageRoute] Pokrycie: {hit}/{total} komorek "
                + $"({covered:F0}%) w promieniu {coverageRadius} m od trasy");

        if (badSegments.Count > 0)
            Debug.LogWarning($"[CoverageRoute] {badSegments.Count} odcinkow nieprzejezdnych "
                           + $"lub przechodzacych przez sciane (indeksy: "
                           + $"{string.Join(", ", badSegments)}) - na gizmach CZERWONE");
        if (sharpTurns.Count > 0)
            Debug.LogWarning($"[CoverageRoute] {sharpTurns.Count} zakretow ostrzejszych niz "
                           + $"{maxTurnAngle} st (waypointy: {string.Join(", ", sharpTurns)}) - "
                           + $"auto moze tam nie wyrobic. Na gizmach POMARANCZOWE");
        if (badSegments.Count == 0 && sharpTurns.Count == 0)
            Debug.Log("[CoverageRoute] Trasa przejezdna, bez ostrych zakretow.");
    }

    private float ComputeCoverage(out int totalCells, out int coveredCells)
    {
        uncoveredCells.Clear();
        totalCells = 0;
        coveredCells = 0;

        NavMeshTriangulation tri = NavMesh.CalculateTriangulation();
        if (tri.indices.Length == 0) return 0f;

        var cells = new Dictionary<Vector2Int, Vector3>();
        for (int t = 0; t < tri.indices.Length; t += 3)
        {
            Vector3 a = tri.vertices[tri.indices[t]];
            Vector3 b = tri.vertices[tri.indices[t + 1]];
            Vector3 c = tri.vertices[tri.indices[t + 2]];
            Vector3 mid = (a + b + c) / 3f;
            var key = new Vector2Int(Mathf.FloorToInt(mid.x / gridCellSize),
                                     Mathf.FloorToInt(mid.z / gridCellSize));
            if (!cells.ContainsKey(key)) cells[key] = mid;
        }

        var p = GetPolyline();
        float r2 = coverageRadius * coverageRadius;
        foreach (var kv in cells)
        {
            totalCells++;
            bool near = false;
            for (int i = 0; i < p.Count - 1 && !near; i++)
            {
                Vector3 cp = ClosestPointOnSegment(Flat(p[i]), Flat(p[i + 1]), Flat(kv.Value));
                if ((cp - Flat(kv.Value)).sqrMagnitude <= r2) near = true;
            }
            if (near) coveredCells++;
            else uncoveredCells.Add(kv.Value);
        }
        return totalCells > 0 ? 100f * coveredCells / totalCells : 0f;
    }

    [ContextMenu("Dodaj waypoint na koncu")]
    public void AddWaypoint()
    {
        var go = new GameObject($"wp_{transform.childCount:D2}");
        go.transform.SetParent(transform);
        go.transform.position = transform.childCount > 1
            ? transform.GetChild(transform.childCount - 2).position + Vector3.forward * 2f
            : transform.position;
#if UNITY_EDITOR
        Undo.RegisterCreatedObjectUndo(go, "Dodaj waypoint");
        Selection.activeGameObject = go;
#endif
    }

    // =======================================================================

    private static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

    private static Vector3 ClosestPointOnSegment(Vector3 a, Vector3 b, Vector3 p)
    {
        Vector3 ab = b - a;
        float len2 = ab.sqrMagnitude;
        if (len2 < 1e-6f) return a;
        return a + ab * Mathf.Clamp01(Vector3.Dot(p - a, ab) / len2);
    }

    void OnDrawGizmos()
    {
        if (!showGizmos) return;
        var p = GetPolyline();
        if (p.Count < 2) return;

        for (int i = 0; i < p.Count - 1; i++)
        {
            Gizmos.color = badSegments.Contains(i) ? Color.red : Color.cyan;
            Vector3 a = p[i] + Vector3.up * 0.1f, b = p[i + 1] + Vector3.up * 0.1f;
            Gizmos.DrawLine(a, b);

            // strzalka kierunku jazdy w polowie odcinka
            Vector3 mid = (a + b) * 0.5f;
            Vector3 dir = (b - a).normalized;
            Vector3 side = Vector3.Cross(dir, Vector3.up) * 0.25f;
            Gizmos.DrawLine(mid, mid - dir * 0.5f + side);
            Gizmos.DrawLine(mid, mid - dir * 0.5f - side);
        }

        for (int i = 0; i < p.Count - (closedLoop ? 1 : 0); i++)
        {
            Gizmos.color = sharpTurns.Contains(i) ? new Color(1f, 0.5f, 0f) : Color.yellow;
            Gizmos.DrawWireSphere(p[i] + Vector3.up * 0.1f, 0.3f);
#if UNITY_EDITOR
            if (labelEvery > 0 && (i % labelEvery == 0 || sharpTurns.Contains(i)))
                Handles.Label(p[i] + Vector3.up * 0.4f, i.ToString());
#endif
        }

        if (showCoverage)
        {
            Gizmos.color = new Color(1f, 0f, 0f, 0.3f);
            foreach (var c in uncoveredCells)
                Gizmos.DrawCube(c + Vector3.up * 0.05f,
                                new Vector3(gridCellSize * 0.8f, 0.02f, gridCellSize * 0.8f));
        }
    }
}