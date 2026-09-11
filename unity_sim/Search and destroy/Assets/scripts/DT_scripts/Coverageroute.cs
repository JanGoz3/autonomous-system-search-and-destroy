using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
#if UNITY_EDITOR
using UnityEditor;
#endif

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
    [Tooltip("Kat zakretu powyzej ktorego ostrzegamy. 30 st, nie 100: przy progu 100 rog 90 st "
           + "przechodzil walidacje, a wlasnie na nim auto scinalo zakret, wpadalo w sciane i "
           + "pursuit point zostawal ZA nim. Uzyj kontekstowego Zaokraglij ostre narozniki.")]
    public float maxTurnAngle = 30f;
    [Tooltip("Jesli sciezka NavMesh miedzy waypointami jest dluzsza niz tyle razy odleglosc w linii prostej, odcinek przechodzi przez sciane.")]
    public float maxDetourRatio = 1.4f;

    [Header("Zaokraglanie narozników")]
    [Tooltip("Promien zaokraglenia. UWAGA na kompromis: luk przesuwa trase do WNETRZA "
           + "zakretu o okolo 0.35 * cornerRadius. Przy 1.3 m to 46 cm, czyli w korytarzu "
           + "szerokosci 2 m polowe odleglosci do wewnetrznej sciany - auto zacznie ja "
           + "ocierac. Zacznij od 0.6 i zwiekszaj po 0.2, sprawdzajac jazde po kazdym kroku. "
           + "lookAheadDistance MUSI byc <= cornerRadius, inaczej punkt poscigowy przeskakuje "
           + "nad lukiem i auto scina zakret mocniej niz przed zaokragleniem.")]
    public float cornerRadius = 0.6f;
    [Tooltip("Ile punktow wstawic na kazdym zaokragleniu.")]
    public int cornerSegments = 4;
    [Tooltip("Zaokraglaj tylko zakrety ostrzejsze niz tyle stopni.")]
    public float filletThresholdDeg = 30f;

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

    [Header("Kopia zapasowa")]
    [Tooltip("Pozycje waypointow zapisane przed ostatnia operacja niszczaca. "
           + "Serializowane, wiec przezywaja zapis sceny i restart edytora.")]
    [SerializeField] private List<Vector3> routeBackup = new List<Vector3>();
    [SerializeField] private string routeBackupNote = "";

    private LineRenderer preview;

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

    public void InvalidateCache() => cacheFrame = -1;

    public void ReverseDirection()
    {
        var kids = new List<Transform>();
        foreach (Transform c in transform)
        {
            if (c.name == "__routePreview") continue;
            kids.Add(c);
        }
        if (kids.Count < 2) return;

        var pts = new List<Vector3>(kids.Count);
        for (int i = 0; i < kids.Count; i++) pts.Add(kids[i].position);
        pts.Reverse();
        for (int i = 0; i < kids.Count; i++) kids[i].position = pts[i];

        InvalidateCache();
        if (useLineRenderer && preview != null) BuildPreview();
    }

    [ContextMenu("Odwroc kierunek trasy")]
    private void ReverseDirectionMenu()
    {
        ReverseDirection();
        Debug.Log("[CoverageRoute] Odwrocono kierunek trasy.");
    }

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

    public Vector3 PointAtDistance(float d)
    {
        EnsureCache();
        if (cachedPts.Count < 2) return transform.position;

        if (closedLoop && cachedLength > 1e-3f) d = Mathf.Repeat(d, cachedLength);
        d = Mathf.Clamp(d, 0f, cachedLength);

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

        int last = closedLoop ? p.Count - 1 : p.Count - 1;
        for (int i = 1; i < last; i++)
        {
            Vector3 inDir = Flat(p[i] - p[i - 1]).normalized;
            Vector3 outDir = Flat(p[i + 1] - p[i]).normalized;
            if (inDir.sqrMagnitude < 0.1f || outDir.sqrMagnitude < 0.1f) continue;
            if (Vector3.Angle(inDir, outDir) > maxTurnAngle) sharpTurns.Add(i);
        }

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


    [ContextMenu("Zapisz kopie trasy")]
    public void SaveRouteBackupManual() => SaveRouteBackup("recznie");

    private void SaveRouteBackup(string note)
    {
        routeBackup.Clear();
        foreach (Transform c in transform)
        {
            if (c.name == "__routePreview") continue;
            routeBackup.Add(c.position);
        }
        routeBackupNote = $"{routeBackup.Count} pkt, {note}, {System.DateTime.Now:HH:mm:ss}";
        Debug.Log($"[CoverageRoute] Kopia zapasowa: {routeBackupNote}");
    }

    [ContextMenu("Przywroc trase z kopii")]
    public void RestoreRouteBackup()
    {
        if (routeBackup == null || routeBackup.Count < 2)
        {
            Debug.LogWarning("[CoverageRoute] Brak kopii zapasowej do przywrocenia.");
            return;
        }

        var pts = new List<Vector3>(routeBackup);   // kopia: przebudowa czysci pole
#if UNITY_EDITOR
        Undo.RegisterFullObjectHierarchyUndo(gameObject, "Przywroc trase");
#endif
        ClearPreview();
        for (int i = transform.childCount - 1; i >= 0; i--)
            DestroyImmediate(transform.GetChild(i).gameObject);

        for (int i = 0; i < pts.Count; i++)
        {
            var go = new GameObject($"wp_{i:D3}");
            go.transform.SetParent(transform);
            go.transform.position = pts[i];
        }

        InvalidateCache();
        if (useLineRenderer) BuildPreview();
        Debug.Log($"[CoverageRoute] Przywrocono {pts.Count} waypointow ({routeBackupNote}).");
    }

    [Header("Upraszczanie")]
    [Tooltip("Tolerancja Ramer-Douglas-Peucker w metrach. Punkty odchylajace sie od "
           + "uproszczonej lamanej mniej niz tyle sa usuwane. 0.35 m zwija typowe "
           + "zaokraglenie z powrotem w ostry narozník; wieksze wartosci zaczna "
           + "wycinac prawdziwe zakrety.")]
    public float simplifyToleranceM = 0.35f;

    /// <summary>Usuwa nadmiarowe waypointy algorytmem Ramer-Douglas-Peucker.
    /// Sluzy do ODWROCENIA zaokraglenia, gdy kopia zapasowa nie jest dostepna:
    /// luki zwijaja sie z powrotem do narozníkow, bo ich punkty odchylaja sie od
    /// lamanej mniej niz tolerancja. Nie odtwarza pozycji bit w bit, ale wraca do
    /// ksztaltu sprzed zaokraglenia.</summary>
    [ContextMenu("Uprosc trase (odwroc zaokraglenie)")]
    public void SimplifyRoute()
    {
        var src = new List<Vector3>();
        foreach (Transform c in transform)
        {
            if (c.name == "__routePreview" || !c.gameObject.activeSelf) continue;
            src.Add(c.position);
        }
        if (src.Count < 4)
        {
            Debug.LogWarning("[CoverageRoute] Za malo waypointow do uproszczenia.");
            return;
        }

        var keep = new bool[src.Count];
        keep[0] = keep[src.Count - 1] = true;
        RdpRecurse(src, 0, src.Count - 1, simplifyToleranceM, keep);

        var outPts = new List<Vector3>();
        for (int i = 0; i < src.Count; i++) if (keep[i]) outPts.Add(src[i]);

        if (outPts.Count == src.Count)
        {
            Debug.Log($"[CoverageRoute] Nic nie usunieto przy tolerancji "
                    + $"{simplifyToleranceM} m. Zwieksz simplifyToleranceM.");
            return;
        }

#if UNITY_EDITOR
        Undo.RegisterFullObjectHierarchyUndo(gameObject, "Uprosc trase");
#endif
        SaveRouteBackup($"przed uproszczeniem (tol={simplifyToleranceM})");

        ClearPreview();
        for (int i = transform.childCount - 1; i >= 0; i--)
            DestroyImmediate(transform.GetChild(i).gameObject);

        for (int i = 0; i < outPts.Count; i++)
        {
            var go = new GameObject($"wp_{i:D3}");
            go.transform.SetParent(transform);
            go.transform.position = outPts[i];
        }

        InvalidateCache();
        if (useLineRenderer) BuildPreview();
        Debug.Log($"[CoverageRoute] Uproszczono: {src.Count} -> {outPts.Count} waypointow. "
                + "Uruchom Waliduj trase i sprawdz jazde eksperta.");
    }

    private static void RdpRecurse(List<Vector3> pts, int lo, int hi, float tol, bool[] keep)
    {
        if (hi <= lo + 1) return;

        Vector3 a = Flat(pts[lo]), b = Flat(pts[hi]);
        float worst = -1f;
        int worstIdx = -1;

        for (int i = lo + 1; i < hi; i++)
        {
            Vector3 p = Flat(pts[i]);
            float d = (p - ClosestPointOnSegment(a, b, p)).magnitude;
            if (d > worst) { worst = d; worstIdx = i; }
        }

        if (worst > tol && worstIdx > 0)
        {
            keep[worstIdx] = true;
            RdpRecurse(pts, lo, worstIdx, tol, keep);
            RdpRecurse(pts, worstIdx, hi, tol, keep);
        }
    }

    [ContextMenu("Zaokraglij ostre narozniki")]
    public void FilletCorners()
    {
        var src = new List<Vector3>();
        foreach (Transform c in transform)
        {
            if (c.name == "__routePreview" || !c.gameObject.activeSelf) continue;
            src.Add(c.position);
        }
        if (src.Count < 3)
        {
            Debug.LogWarning("[CoverageRoute] Za malo waypointow do zaokraglenia (min. 3).");
            return;
        }

        var outPts = new List<Vector3>();
        int n = src.Count;
        int filleted = 0;

        for (int i = 0; i < n; i++)
        {
            Vector3 prev = src[(i - 1 + n) % n];
            Vector3 cur = src[i];
            Vector3 next = src[(i + 1) % n];

            // przy trasie otwartej koncow nie zaokraglamy
            if (!closedLoop && (i == 0 || i == n - 1)) { outPts.Add(cur); continue; }

            Vector3 inV = Flat(cur - prev);
            Vector3 outV = Flat(next - cur);
            if (inV.sqrMagnitude < 1e-4f || outV.sqrMagnitude < 1e-4f)
            { outPts.Add(cur); continue; }

            if (Vector3.Angle(inV, outV) < filletThresholdDeg) { outPts.Add(cur); continue; }

            // przyciecie ograniczone do 45% krotszego odcinka, zeby zaokraglenia
            // sasiednich narozników nie zjadly sie wzajemnie
            float d = Mathf.Min(cornerRadius, 0.45f * inV.magnitude, 0.45f * outV.magnitude);
            Vector3 a = cur - inV.normalized * d;
            Vector3 b = cur + outV.normalized * d;

            int seg = Mathf.Max(2, cornerSegments);
            for (int k = 0; k <= seg; k++)
            {
                float t = (float)k / seg;
                outPts.Add(Vector3.Lerp(Vector3.Lerp(a, cur, t), Vector3.Lerp(cur, b, t), t));
            }
            filleted++;
        }

        if (filleted == 0)
        {
            Debug.Log($"[CoverageRoute] Brak zakretow ostrzejszych niz "
                    + $"{filletThresholdDeg} st - nic nie zmieniono.");
            return;
        }

#if UNITY_EDITOR
        Undo.RegisterFullObjectHierarchyUndo(gameObject, "Zaokraglij narozniki");
#endif
        // Kopia PRZED zniszczeniem dzieci. Ctrl+Z dziala tylko bezposrednio po
        // operacji, wiec bez tego jedna nieuwazna akcja kosztuje cala trase.
        SaveRouteBackup($"przed zaokragleniem (r={cornerRadius}, prog={filletThresholdDeg})");

        ClearPreview();
        for (int i = transform.childCount - 1; i >= 0; i--)
            DestroyImmediate(transform.GetChild(i).gameObject);

        for (int i = 0; i < outPts.Count; i++)
        {
            var go = new GameObject($"wp_{i:D3}");
            go.transform.SetParent(transform);
            go.transform.position = outPts[i];
        }

        InvalidateCache();
        if (useLineRenderer) BuildPreview();

        Debug.Log($"[CoverageRoute] Zaokraglono {filleted} narozników: "
                + $"{n} -> {outPts.Count} waypointow. Uruchom teraz Waliduj trase - "
                + "nie moze zglosic ani jednego ostrego zakretu.");
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