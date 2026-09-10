using UnityEngine;
using UnityEngine.AI;


public class AutoExplorer : MonoBehaviour
{
    [Header("References")]
    public CoverageRoute route;
    public Transform carTransform;
    public Rigidbody carRigidbody;
    public Transform target;
    [Tooltip("Opcjonalne. Jesli podpiete, zaklinowanie KONCZY epizod (logger zapisuje fragment albo odrzuca go, gdy za krotki) i dopiero potem auto jest przenoszone.")]
    public DTDataLogger dataLogger;
    [Tooltip("Opcjonalne, ale ZALECANE. Bufor ToF indeksuje sektory yawem WZGLEDEM AUTA i trzyma pomiary przez maxMeasurementAgeSeconds. Po teleportacji te pomiary opisuja poprzednie miejsce - nie sa 'stare', tylko FALSZYWE.")]
    public TofScanBuffer tofScanBuffer;

    [Header("Pure pursuit")]
    [Tooltip("Jak daleko przed autem trzymac target, mierzone WZDLUZ SCIEZKI NavMesh. Ustaw rowno z WAYPOINT_DIST w build_dt_dataset.py.")]
    public float lookAheadDistance = 1.5f;
    [Tooltip("Jak daleko do przodu szukac rzutu auta na trase przy kazdej klatce.")]
    public float searchForward = 6f;
    [Tooltip("Ile wstecz. Male, zeby auto nie zrzutowalo sie na wczesniejszy fragment trasy w rownoleglym korytarzu.")]
    public float searchBackward = 1f;
    public float searchStep = 0.2f;
    [Tooltip("O ile metrow ponad faktycznie przejechany dystans postep moze wzrosnac w jednej klatce.")]
    public float progressSlack = 0.3f;
    [Tooltip("Powyzej tego odchylenia od trasy szukamy rzutu po CALEJ trasie, nie tylko w oknie.")]
    public float maxDeviation = 4f;

    [Header("Sciezka NavMesh (etykieta)")]
    [Tooltip("Gdy true, etykieta jest liczona wzdluz sciezki NavMesh - ekspert obchodzi sciany "
           + "i wskazuje drzwi. Gdy false, wraca stare zachowanie (punkt na trasie w linii "
           + "prostej). Wylaczaj tylko do porownania ze starym datasetem.")]
    public bool useNavMeshPath = true;
    [Tooltip("Co ile sekund przeliczac sciezke. Punkt docelowy w SWIECIE jest stabilny, wiec "
           + "nie trzeba go liczyc co klatke - konwersja do ukladu auta i tak dzieje sie zawsze. "
           + "0.05 = 20 Hz, dwa razy czesciej niz logowanie.")]
    public float pathUpdateInterval = 0.05f;
    [Tooltip("Promien, w jakim szukamy najblizszego punktu NavMesh dla pozycji auta i celu.")]
    public float navSampleRadius = 1.5f;

    [Header("Tryb pracy")]
    [Tooltip("Gdy false, AutoExplorer LICZY expertLocalWaypoint, ale NIE rusza obiektu target. "
           + "Uzywane przez CoverageBenchmark (polityka ExpertFrozen) i przy DAggerze.")]
    public bool driveTarget = true;

    [Header("Spawn")]
    [Tooltip("Start w losowym punkcie trasy - daje zroznicowane pozycje poczatkowe.")]
    public bool respawnOnStart = true;
    [Tooltip("Losowe odchylenie od kierunku trasy przy spawnie, w stopniach (+/-). Przy duzych wartosciach co drugi respawn zaczyna sie od zawracania, ktore detektor bierze za zaklinowanie.")]
    public float spawnYawJitter = 50f;
    public float spawnHeightOffset = 0.2f;

    [Header("Utkniecie")]
    public bool detectStuck = true;
    [Tooltip("Przez ile sekund auto musi nie ruszyc sie o stuckDistanceThreshold, zeby uznac je za zaklinowane.")]
    public float stuckCheckWindow = 6f;
    public float stuckDistanceThreshold = 0.5f;
    [Tooltip("Karencja po respawnie - przez tyle sekund nie sprawdzamy zaklinowania.")]
    public float spawnGracePeriod = 5f;
    [Tooltip("Po wykryciu zaklinowania: zakoncz epizod w loggerze i przenies auto w losowy punkt trasy.")]
    public bool respawnWhenStuck = true;

    [Header("Runtime (read-only)")]
    public bool isExploring = false;
    public float progressAlongRoute = 0f;
    public float deviationFromRoute = 0f;
    public int lapsCompleted = 0;
    public int stuckEvents = 0;
    [Tooltip("Postep na trasie przy kolejnych zaklinowaniach. Skupienie wartosci = konkretne zle miejsce na trasie.")]
    public string stuckHotspots = "";
    [Tooltip("ETYKIETA dla DT: wektor do pursuit pointa w ukladzie auta. x = w prawo, z = do przodu, w metrach.")]
    public Vector2 expertLocalWaypoint;
    [Tooltip("Pursuit point w ukladzie SWIATA. Liczony zawsze, niezaleznie od driveTarget.")]
    public Vector3 expertWorldWaypointRaw;
    [Tooltip("Czy udalo sie wyznaczyc PELNA sciezke NavMesh do punktu na trasie. Gdy false, "
           + "ekspert nie wie, jak tam dojechac, i etykieta jest fallbackiem w linii prostej - "
           + "takie klatki warto odfiltrowac z lossu (kolumna expert_valid).")]
    public bool expertPathValid = false;
    [Tooltip("Dlugosc sciezki NavMesh do punktu na trasie. Duzo wieksza od odleglosci w linii "
           + "prostej = auto jest po drugiej stronie sciany i musi obchodzic.")]
    public float pathLengthToGoal = 0f;
    [Tooltip("Udzial klatek, w ktorych sciezka byla niepelna. Wysokie wartosci = trasa wychodzi "
           + "poza NavMesh albo auto ciagle laduje w miejscach bez dojazdu.")]
    public float pathFailRate = 0f;

    private float stuckTimer = 0f;
    private float graceTimer = 0f;
    private Vector3 stuckAnchor;
    private float routeLength = 0f;
    private Vector3 lastProjectionPos;
    private readonly System.Collections.Generic.List<float> stuckAt =
        new System.Collections.Generic.List<float>();

    private NavMeshPath navPath;
    private float pathTimer = 0f;
    private int pathCalls = 0, pathFails = 0;

    public Vector3 ExpertWorldWaypoint => expertWorldWaypointRaw;
    public bool StuckThisFrame { get; private set; }


    void Reset()
    {
        carTransform = transform;
        carRigidbody = GetComponent<Rigidbody>();
        dataLogger = GetComponent<DTDataLogger>();
        tofScanBuffer = GetComponentInChildren<TofScanBuffer>(true);
    }

    void Awake()
    {
        navPath = new NavMeshPath();
    }

    public void StartExploring()
    {
        if (route == null || carTransform == null || target == null)
        {
            Debug.LogError("[AutoExplorer] Brak referencji route/carTransform/target.");
            return;
        }

        routeLength = route.TotalLength();
        if (routeLength < 1f)
        {
            Debug.LogError("[AutoExplorer] Trasa pusta lub za krotka.");
            return;
        }

        if (navPath == null) navPath = new NavMeshPath();
        if (tofScanBuffer == null)
            Debug.LogWarning("[AutoExplorer] Brak referencji TofScanBuffer. Po respawnie bufor "
                + "nie zostanie wyczyszczony i bedzie opisywal poprzednie miejsce.", this);

        if (respawnOnStart) RespawnOnRoute();
        else progressAlongRoute = ProjectGlobally();

        stuckTimer = 0f;
        graceTimer = spawnGracePeriod;
        stuckAnchor = carTransform.position;
        lastProjectionPos = carTransform.position;
        stuckEvents = 0;
        stuckAt.Clear();
        stuckHotspots = "";
        lapsCompleted = 0;
        pathCalls = 0;
        pathFails = 0;
        pathFailRate = 0f;
        pathTimer = 0f;
        isExploring = true;

        RecomputeWorldWaypoint();
        UpdateTarget();
        Debug.Log($"[AutoExplorer] Start. Trasa {routeLength:F1} m, postep {progressAlongRoute:F1} m, "
                + $"etykieta {(useNavMeshPath ? "WZDLUZ SCIEZKI NavMesh" : "w linii prostej")}");
    }

    public void StopExploring()
    {
        isExploring = false;
        if (pathCalls > 0)
            Debug.Log($"[AutoExplorer] Sciezka NavMesh: {pathFails}/{pathCalls} nieudanych "
                    + $"({100f * pathFails / pathCalls:F1}%)");
    }

    public void RespawnOnRoute()
    {
        float d = Random.Range(0f, routeLength);
        Vector3 pos = route.PointAtDistance(d);

        Vector3 ahead = route.PointAtDistance(d + 1f);
        Quaternion rot = Quaternion.LookRotation(
            Flat(ahead - pos).sqrMagnitude > 1e-4f ? Flat(ahead - pos) : Vector3.forward,
            Vector3.up);
        if (spawnYawJitter > 0f)
            rot *= Quaternion.Euler(0f, Random.Range(-spawnYawJitter, spawnYawJitter), 0f);

        if (NavMesh.SamplePosition(pos, out NavMeshHit hit, 2f, NavMesh.AllAreas))
            pos = hit.position;

        carTransform.SetPositionAndRotation(pos + Vector3.up * spawnHeightOffset, rot);
        if (carRigidbody != null)
        {
            carRigidbody.linearVelocity = Vector3.zero;
            carRigidbody.angularVelocity = Vector3.zero;
        }

        if (tofScanBuffer != null) tofScanBuffer.Clear();

        progressAlongRoute = d;
        stuckAnchor = carTransform.position;
        lastProjectionPos = carTransform.position;
        stuckTimer = 0f;
        graceTimer = spawnGracePeriod;
        pathTimer = 0f;
    }

    void Update()
    {
        if (!isExploring) return;

        UpdateProgress();
        CheckStuck();

        pathTimer -= Time.deltaTime;
        if (pathTimer <= 0f)
        {
            pathTimer = Mathf.Max(0.01f, pathUpdateInterval);
            RecomputeWorldWaypoint();
        }

        UpdateTarget();
    }

    private void UpdateProgress()
    {
        float best = float.MaxValue, bestD = progressAlongRoute;
        Vector3 car = Flat(carTransform.position);

        for (float d = progressAlongRoute - searchBackward;
             d <= progressAlongRoute + searchForward; d += searchStep)
        {
            float sq = (Flat(route.PointAtDistance(d)) - car).sqrMagnitude;
            if (sq < best) { best = sq; bestD = d; }
        }

        deviationFromRoute = Mathf.Sqrt(best);

        if (deviationFromRoute > maxDeviation)
        {
            bestD = ProjectGlobally();
            deviationFromRoute = (Flat(route.PointAtDistance(bestD)) - car).magnitude;
            lastProjectionPos = carTransform.position;
        }
        else
        {
            float travelled = Vector3.Distance(car, Flat(lastProjectionPos));
            bestD = Mathf.Min(bestD, progressAlongRoute + travelled + progressSlack);
            lastProjectionPos = carTransform.position;
        }

        if (routeLength > 1f && Mathf.FloorToInt(bestD / routeLength)
                              > Mathf.FloorToInt(progressAlongRoute / routeLength))
            lapsCompleted++;

        progressAlongRoute = Mathf.Max(progressAlongRoute, bestD);
    }

    private float ProjectGlobally()
    {
        float best = float.MaxValue, bestD = 0f;
        Vector3 car = Flat(carTransform.position);
        for (float d = 0f; d < routeLength; d += searchStep * 2f)
        {
            float sq = (Flat(route.PointAtDistance(d)) - car).sqrMagnitude;
            if (sq < best) { best = sq; bestD = d; }
        }
        return bestD;
    }

    private void RecomputeWorldWaypoint()
    {

        Vector3 goal = route.PointAtDistance(progressAlongRoute + lookAheadDistance);

        if (!useNavMeshPath)
        {
            expertWorldWaypointRaw = goal;
            expertPathValid = true;
            pathLengthToGoal = Vector3.Distance(Flat(carTransform.position), Flat(goal));
            return;
        }

        pathCalls++;
        expertPathValid = false;
        Vector3 result = goal;

        if (NavMesh.SamplePosition(carTransform.position, out NavMeshHit fromHit,
                                   navSampleRadius, NavMesh.AllAreas)
            && NavMesh.SamplePosition(goal, out NavMeshHit goalHit,
                                      navSampleRadius, NavMesh.AllAreas)
            && NavMesh.CalculatePath(fromHit.position, goalHit.position,
                                     NavMesh.AllAreas, navPath)
            && navPath.status == NavMeshPathStatus.PathComplete
            && navPath.corners.Length >= 2)
        {
            expertPathValid = true;
            result = PointAlongPath(navPath.corners, lookAheadDistance, out pathLengthToGoal);
        }
        else
        {
            pathFails++;
            pathLengthToGoal = Vector3.Distance(Flat(carTransform.position), Flat(goal));
        }

        pathFailRate = pathCalls > 0 ? (float)pathFails / pathCalls : 0f;
        expertWorldWaypointRaw = result;
    }

    private static Vector3 PointAlongPath(Vector3[] corners, float dist, out float totalLength)
    {
        totalLength = 0f;
        for (int i = 0; i < corners.Length - 1; i++)
            totalLength += Vector3.Distance(Flat(corners[i]), Flat(corners[i + 1]));

        float remaining = dist;
        for (int i = 0; i < corners.Length - 1; i++)
        {
            float seg = Vector3.Distance(Flat(corners[i]), Flat(corners[i + 1]));
            if (seg < 1e-4f) continue;
            if (remaining <= seg)
                return Vector3.Lerp(corners[i], corners[i + 1], remaining / seg);
            remaining -= seg;
        }
        return corners[corners.Length - 1];
    }

    private void UpdateTarget()
    {
        Vector3 local = Quaternion.Inverse(
            Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f))
            * Flat(expertWorldWaypointRaw - carTransform.position);
        expertLocalWaypoint = new Vector2(local.x, local.z);

        if (driveTarget)
            target.position = expertWorldWaypointRaw + new Vector3(0, 0.15f, 0);
    }

    private void CheckStuck()
    {
        StuckThisFrame = false;
        if (!detectStuck) return;

        if (graceTimer > 0f)
        {
            graceTimer -= Time.deltaTime;
            stuckTimer = 0f;
            stuckAnchor = carTransform.position;
            return;
        }

        stuckTimer += Time.deltaTime;
        if (stuckTimer < stuckCheckWindow) return;
        stuckTimer = 0f;

        if (Vector3.Distance(Flat(carTransform.position), Flat(stuckAnchor))
            < stuckDistanceThreshold)
        {
            stuckEvents++;
            StuckThisFrame = true;
            RecordHotspot(progressAlongRoute);

            if (respawnWhenStuck)
            {
                // KOLEJNOSC JEST ISTOTNA: najpierw zamykamy epizod, dopiero potem
                // przenosimy auto - inaczej teleportacja trafilaby do buforu.
                if (dataLogger != null && dataLogger.isRecording)
                    dataLogger.RestartEpisode($"zaklinowanie na {progressAlongRoute:F1} m trasy");
                RespawnOnRoute();
            }
        }
        stuckAnchor = carTransform.position;
    }

    private void RecordHotspot(float d)
    {
        stuckAt.Add(d);
        if (stuckAt.Count > 40) stuckAt.RemoveAt(0);

        int buckets = Mathf.Max(1, Mathf.CeilToInt(routeLength / 5f));
        var counts = new int[buckets];
        foreach (float x in stuckAt)
            counts[Mathf.Clamp(Mathf.FloorToInt(Mathf.Repeat(x, routeLength) / 5f), 0, buckets - 1)]++;

        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < buckets; i++)
            if (counts[i] > 0) sb.Append($"{i * 5}-{(i + 1) * 5}m:{counts[i]}  ");
        stuckHotspots = sb.ToString();
    }

    private static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

    void OnDrawGizmos()
    {
        if (!isExploring || carTransform == null || target == null) return;

        if (useNavMeshPath && navPath != null && navPath.corners != null
            && navPath.corners.Length >= 2)
        {
            Gizmos.color = expertPathValid ? Color.white : Color.red;
            for (int i = 0; i < navPath.corners.Length - 1; i++)
                Gizmos.DrawLine(navPath.corners[i] + Vector3.up * 0.05f,
                                navPath.corners[i + 1] + Vector3.up * 0.05f);
        }

        Gizmos.color = expertPathValid ? Color.green : Color.red;   // pursuit point EKSPERTA
        Gizmos.DrawWireSphere(expertWorldWaypointRaw + Vector3.up * 0.1f, 0.3f);
        Gizmos.DrawLine(carTransform.position, expertWorldWaypointRaw);

        if (!driveTarget && target != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(target.position + Vector3.up * 0.1f, Vector3.one * 0.3f);
        }

        if (route != null)
        {
            Gizmos.color = Color.magenta;                 // rzut auta na trase
            Gizmos.DrawWireSphere(
                route.PointAtDistance(progressAlongRoute) + Vector3.up * 0.1f, 0.2f);
        }

        Gizmos.color = Color.cyan;                        // kierunek auta
        Gizmos.DrawRay(carTransform.position,
            Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f) * Vector3.forward * 1.5f);
    }
}