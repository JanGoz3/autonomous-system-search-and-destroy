using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// Eksploracja: wybor celu metoda FRONTIER + prowadzenie metoda PRZYNETY.
///
/// Dwie niezalezne warstwy:
///
/// 1. DOKAD (frontier). Z NavMesh.CalculateTriangulation() budujemy raz na
///    starcie zbior wszystkich chodliwych komorek 1x1 m - tej samej siatki,
///    ktora liczy nagrode w build_dt_dataset.py. Celem jest zawsze najblizsza
///    NIEODWIEDZONA komorka osiagalna po NavMeshu.
///
///    Poprzednia wersja losowala punkt B, wiec nie optymalizowala pokrycia w
///    ogole - DT uczyl sie imitowac eksperta, ktory nie byl dobry w zadaniu,
///    za ktore DT dostaje nagrode. Teraz etykiety zawieraja strategie.
///
/// 2. JAK (przyneta). Target trzymany w stalej, malej odleglosci PRZED autem
///    wzdluz wyznaczonej trasy. Pozycja targetu jest funkcja pozycji auta,
///    a nie czasu, wiec nie ma czego dostrajac do predkosci drivera.
///
/// CurrentTargetPosition to sygnal planera - logujac go mozna relabelowac
/// akcje DT z planu zamiast z faktycznej trasy auta.
/// </summary>
public class AutoExplorer : MonoBehaviour
{
    [Header("References")]
    public Transform carTransform;
    public Transform target;

    [Header("Przyneta")]
    [Tooltip("Jak daleko przed autem trzymac target, mierzone wzdluz trasy. Ustaw rowno z WAYPOINT_DIST w build_dt_dataset.py.")]
    public float lookAheadDistance = 1.5f;

    [Header("Frontier")]
    [Tooltip("MUSI rownac sie GRID_CELL_SIZE w build_dt_dataset.py i gridCellSize w DTInference.")]
    public float gridCellSize = 1.0f;
    [Tooltip("Nie wybieraj celow blizszych niz tyle - inaczej auto dreptaloby w miejscu, przeskakujac miedzy sasiednimi komorkami.")]
    public float minTargetDistance = 4f;
    [Tooltip("Ilu najblizszych kandydatow sprawdzic przez CalculatePath, zanim sie poddamy.")]
    public int candidatesToTry = 12;
    [Tooltip("Po pokryciu calej mapy wyczysc historie i zacznij od nowa (tryb patrolu). Wylaczone = przejscie na cele losowe.")]
    public bool restartWhenComplete = true;

    [Header("Trasa")]
    [Tooltip("Gdy do konca trasy zostanie mniej niz tyle metrow, wybieramy nastepny frontier - bez zatrzymywania auta.")]
    public float replanWhenRemaining = 2f;
    [Tooltip("Jesli auto oddali sie od trasy bardziej niz tyle, planujemy od nowa z jego faktycznej pozycji.")]
    public float maxDeviation = 4f;

    [Header("Przeplanowanie przy utknieciu")]
    public bool replanWhenStuck = true;
    public float stuckCheckWindow = 5f;
    public float stuckDistanceThreshold = 0.4f;

    [Header("Runtime (read-only)")]
    public bool isExploring = false;
    public int walkableCellCount = 0;
    public int visitedCellCount = 0;
    public float coveragePercent = 0f;
    public float progressAlongPath = 0f;
    public float pathTotalLength = 0f;
    public float deviationFromPath = 0f;
    public int pathsPlanned = 0;
    public int stuckReplans = 0;
    public int mapRestarts = 0;

    private readonly Dictionary<Vector2Int, Vector3> walkable = new Dictionary<Vector2Int, Vector3>();
    private readonly HashSet<Vector2Int> visited = new HashSet<Vector2Int>();
    private readonly HashSet<Vector2Int> unreachable = new HashSet<Vector2Int>();

    private readonly List<Vector3> corners = new List<Vector3>();
    private readonly List<float> cumulative = new List<float>();
    private NavMeshPath navPath;                 // NIE w inicjalizatorze pola
    private int currentSegment = 0;

    private float stuckTimer = 0f;
    private Vector3 stuckAnchor;

    public Vector3 CurrentTargetPosition => target != null ? target.position : Vector3.zero;
    public bool HasPath => corners.Count >= 2;

    // =======================================================================

    void Awake()
    {
        if (navPath == null) navPath = new NavMeshPath();
    }

    public void StartExploring()
    {
        if (carTransform == null || target == null)
        {
            Debug.LogError("[AutoExplorer] Brak referencji carTransform/target.");
            return;
        }
        if (navPath == null) navPath = new NavMeshPath();

        BuildWalkableCells();
        if (walkable.Count == 0)
        {
            Debug.LogError("[AutoExplorer] NavMesh nie zwrocil zadnych trojkatow. "
                         + "Czy scena ma zbakowany NavMesh?");
            return;
        }

        visited.Clear();
        unreachable.Clear();
        pathsPlanned = 0;
        stuckReplans = 0;
        mapRestarts = 0;
        stuckTimer = 0f;
        stuckAnchor = carTransform.position;

        MarkVisited();
        isExploring = PlanToFrontier();
        if (!isExploring)
            Debug.LogError("[AutoExplorer] Nie udalo sie wyznaczyc pierwszej trasy.");
    }

    public void StopExploring()
    {
        isExploring = false;
        corners.Clear();
        cumulative.Clear();
    }

    void Update()
    {
        if (!isExploring || !HasPath) return;

        MarkVisited();
        UpdateProgress();
        CheckStuck();

        if (deviationFromPath > maxDeviation ||
            pathTotalLength - progressAlongPath < replanWhenRemaining)
        {
            PlanToFrontier();
            if (!HasPath) return;
            UpdateProgress();
        }

        float d = Mathf.Min(progressAlongPath + lookAheadDistance, pathTotalLength);
        target.position = PointAtDistance(d);
    }

    // ================= FRONTIER ============================================

    /// <summary>Rasteryzuje cala siatke NavMesh na komorki gridCellSize.
    /// Robione RAZ na starcie - triangulacja nie zmienia sie w trakcie.</summary>
    private void BuildWalkableCells()
    {
        walkable.Clear();
        NavMeshTriangulation tri = NavMesh.CalculateTriangulation();

        for (int t = 0; t < tri.indices.Length; t += 3)
        {
            Vector3 a = tri.vertices[tri.indices[t]];
            Vector3 b = tri.vertices[tri.indices[t + 1]];
            Vector3 c = tri.vertices[tri.indices[t + 2]];

            // wierzcholki - lapie waskie korytarze, ktorych srodek komorki mija
            AddCell(a); AddCell(b); AddCell(c);

            int minX = Mathf.FloorToInt(Mathf.Min(a.x, b.x, c.x) / gridCellSize);
            int maxX = Mathf.FloorToInt(Mathf.Max(a.x, b.x, c.x) / gridCellSize);
            int minZ = Mathf.FloorToInt(Mathf.Min(a.z, b.z, c.z) / gridCellSize);
            int maxZ = Mathf.FloorToInt(Mathf.Max(a.z, b.z, c.z) / gridCellSize);

            for (int cx = minX; cx <= maxX; cx++)
                for (int cz = minZ; cz <= maxZ; cz++)
                {
                    var cell = new Vector2Int(cx, cz);
                    if (walkable.ContainsKey(cell)) continue;
                    Vector2 center = new Vector2((cx + 0.5f) * gridCellSize,
                                                 (cz + 0.5f) * gridCellSize);
                    if (PointInTriangle(center, a, b, c))
                        walkable[cell] = new Vector3(center.x, (a.y + b.y + c.y) / 3f, center.y);
                }
        }

        walkableCellCount = walkable.Count;
        Debug.Log($"[AutoExplorer] Siatka: {walkableCellCount} chodliwych komorek "
                + $"{gridCellSize}x{gridCellSize} m");
    }

    private void AddCell(Vector3 p)
    {
        var cell = CellOf(p);
        if (!walkable.ContainsKey(cell)) walkable[cell] = p;
    }

    private void MarkVisited()
    {
        if (visited.Add(CellOf(carTransform.position)))
        {
            visitedCellCount = visited.Count;
            coveragePercent = walkableCellCount > 0
                ? 100f * visitedCellCount / walkableCellCount : 0f;
        }
    }

    /// <summary>Najblizsza nieodwiedzona komorka osiagalna po NavMeshu.</summary>
    private bool PlanToFrontier()
    {
        Vector3 car = carTransform.position;
        if (!NavMesh.SamplePosition(car, out NavMeshHit originHit, 2f, NavMesh.AllAreas))
            return false;

        float minSqr = minTargetDistance * minTargetDistance;
        var candidates = new List<KeyValuePair<float, Vector2Int>>();

        foreach (var kv in walkable)
        {
            if (visited.Contains(kv.Key) || unreachable.Contains(kv.Key)) continue;
            float sqr = (Flat(kv.Value) - Flat(car)).sqrMagnitude;
            if (sqr < minSqr) continue;
            candidates.Add(new KeyValuePair<float, Vector2Int>(sqr, kv.Key));
        }

        if (candidates.Count == 0)
        {
            if (restartWhenComplete)
            {
                mapRestarts++;
                Debug.Log($"[AutoExplorer] Cala mapa pokryta ({visitedCellCount} komorek). "
                        + "Czyszcze historie i zaczynam od nowa.");
                visited.Clear();
                unreachable.Clear();
                visitedCellCount = 0;
                coveragePercent = 0f;
                MarkVisited();
                return PlanToFrontier();
            }
            Debug.LogWarning("[AutoExplorer] Brak nieodwiedzonych komorek.");
            return false;
        }

        candidates.Sort((x, y) => x.Key.CompareTo(y.Key));
        int tries = Mathf.Min(candidatesToTry, candidates.Count);

        for (int i = 0; i < tries; i++)
        {
            Vector2Int cell = candidates[i].Value;
            Vector3 goal = walkable[cell];

            if (!NavMesh.SamplePosition(goal, out NavMeshHit goalHit, gridCellSize * 2f,
                                        NavMesh.AllAreas))
            {
                unreachable.Add(cell);
                continue;
            }
            if (!NavMesh.CalculatePath(originHit.position, goalHit.position,
                                       NavMesh.AllAreas, navPath)
                || navPath.status != NavMeshPathStatus.PathComplete
                || navPath.corners.Length < 2)
            {
                unreachable.Add(cell);   // czarna lista, zeby nie probowac w kolko
                continue;
            }

            StorePath();
            return true;
        }

        // Wszyscy najblizsi kandydaci odpadli - w nastepnej klatce sprobujemy
        // kolejnych, bo trafili na czarna liste.
        return HasPath;
    }

    private void StorePath()
    {
        corners.Clear();
        cumulative.Clear();
        float acc = 0f;
        for (int i = 0; i < navPath.corners.Length; i++)
        {
            if (i > 0)
                acc += Vector3.Distance(Flat(navPath.corners[i - 1]), Flat(navPath.corners[i]));
            corners.Add(navPath.corners[i]);
            cumulative.Add(acc);
        }
        pathTotalLength = acc;
        progressAlongPath = 0f;
        currentSegment = 0;
        deviationFromPath = 0f;
        pathsPlanned++;
    }

    // ================= PRZYNETA ============================================

    /// <summary>Rzutuje auto na trase. Postep NIGDY sie nie cofa - inaczej auto
    /// w rownoleglym korytarzu zrzutowaloby sie na wczesniejszy fragment trasy
    /// i przyneta cofnelaby sie za nie.</summary>
    private void UpdateProgress()
    {
        Vector3 car = Flat(carTransform.position);
        int last = corners.Count - 2;
        int from = Mathf.Clamp(currentSegment, 0, last);
        int to = Mathf.Clamp(currentSegment + 8, 0, last);

        float bestDist = float.MaxValue, bestArc = progressAlongPath;
        int bestSeg = currentSegment;

        for (int i = from; i <= to; i++)
        {
            Vector3 a = Flat(corners[i]), b = Flat(corners[i + 1]);
            Vector3 p = ClosestPointOnSegment(a, b, car);
            float dist = Vector3.Distance(car, p);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestArc = cumulative[i] + Vector3.Distance(a, p);
                bestSeg = i;
            }
        }

        deviationFromPath = bestDist;
        if (bestArc > progressAlongPath)
        {
            progressAlongPath = bestArc;
            currentSegment = bestSeg;
        }
    }

    private void CheckStuck()
    {
        if (!replanWhenStuck) return;
        stuckTimer += Time.deltaTime;
        if (stuckTimer < stuckCheckWindow) return;
        stuckTimer = 0f;

        if (Vector3.Distance(Flat(carTransform.position), Flat(stuckAnchor))
            < stuckDistanceThreshold)
        {
            stuckReplans++;
            PlanToFrontier();
        }
        stuckAnchor = carTransform.position;
    }

    private Vector3 PointAtDistance(float d)
    {
        d = Mathf.Clamp(d, 0f, pathTotalLength);
        for (int i = 0; i < corners.Count - 1; i++)
        {
            if (d <= cumulative[i + 1])
            {
                float segLen = cumulative[i + 1] - cumulative[i];
                float t = segLen > 1e-4f ? (d - cumulative[i]) / segLen : 0f;
                return Vector3.Lerp(corners[i], corners[i + 1], t);
            }
        }
        return corners[corners.Count - 1];
    }

    // ================= pomocnicze ==========================================

    private Vector2Int CellOf(Vector3 p) => new Vector2Int(
        Mathf.FloorToInt(p.x / gridCellSize), Mathf.FloorToInt(p.z / gridCellSize));

    private static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

    private static Vector3 ClosestPointOnSegment(Vector3 a, Vector3 b, Vector3 p)
    {
        Vector3 ab = b - a;
        float len2 = ab.sqrMagnitude;
        if (len2 < 1e-6f) return a;
        return a + ab * Mathf.Clamp01(Vector3.Dot(p - a, ab) / len2);
    }

    private static bool PointInTriangle(Vector2 p, Vector3 a, Vector3 b, Vector3 c)
    {
        float Sign(Vector2 p1, Vector2 p2, Vector2 p3) =>
            (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);

        Vector2 a2 = new Vector2(a.x, a.z), b2 = new Vector2(b.x, b.z), c2 = new Vector2(c.x, c.z);
        float d1 = Sign(p, a2, b2), d2 = Sign(p, b2, c2), d3 = Sign(p, c2, a2);
        bool neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(neg && pos);
    }

    void OnDrawGizmos()
    {
        if (!isExploring) return;

        // nieodwiedzone komorki - to jest "frontier"
        Gizmos.color = new Color(0.2f, 0.6f, 1f, 0.25f);
        foreach (var kv in walkable)
            if (!visited.Contains(kv.Key))
                Gizmos.DrawCube(kv.Value + Vector3.up * 0.05f,
                                new Vector3(gridCellSize * 0.85f, 0.02f, gridCellSize * 0.85f));

        if (corners.Count >= 2)
        {
            Gizmos.color = Color.yellow;
            for (int i = 0; i < corners.Count - 1; i++)
                Gizmos.DrawLine(corners[i] + Vector3.up * 0.1f, corners[i + 1] + Vector3.up * 0.1f);

            Gizmos.color = Color.magenta;
            Gizmos.DrawWireSphere(PointAtDistance(progressAlongPath) + Vector3.up * 0.1f, 0.2f);
        }

        if (target != null && carTransform != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireSphere(target.position + Vector3.up * 0.1f, 0.3f);
            Gizmos.DrawLine(carTransform.position, target.position);
        }
    }
}