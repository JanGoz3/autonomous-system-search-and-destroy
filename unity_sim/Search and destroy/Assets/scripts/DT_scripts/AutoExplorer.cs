using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// Pure pursuit po recznie wyznaczonej trasie CoverageRoute.
///
/// Zastepuje wersje z losowymi punktami B i wersje frontierowa. Roznica jest
/// zasadnicza: trasa jest DETERMINISTYCZNA, wiec etykieta dla DT staje sie
/// funkcja wylacznie tego, co model ma w stanie - pozycji i yaw. Przy losowym
/// punkcie B ten sam stan mial rozne poprawne odpowiedzi zaleznie od tego, co
/// akurat wylosowalo, i to bylo zrodlem rozrzutu 50 st u najblizszych sasiadow.
///
/// ExpertLocalWaypoint to sygnal planera i to jest przyszla ETYKIETA dla DT -
/// zamiast relabelowac z przyszlej pozycji auta (czyli z tego, co udalo sie
/// zrobic driverowi), logujemy to, co ekspert kazal zrobic.
///
/// Zachowany interfejs StartExploring/StopExploring, wiec DTDataLogger
/// i CoverageBenchmark dzialaja bez zmian.
/// </summary>
public class AutoExplorer : MonoBehaviour
{
    [Header("References")]
    public CoverageRoute route;
    public Transform carTransform;
    public Rigidbody carRigidbody;
    public Transform target;
    [Tooltip("Opcjonalne. Jesli podpiete, zaklinowanie KONCZY epizod (logger zapisuje fragment albo odrzuca go, gdy za krotki) i dopiero potem auto jest przenoszone. Bez tego respawn tworzylby w danych falszywa 'teleportacje' w srodku trajektorii.")]
    public DTDataLogger dataLogger;

    [Header("Pure pursuit")]
    [Tooltip("Jak daleko przed autem trzymac target, mierzone wzdluz trasy. Ustaw rowno z WAYPOINT_DIST w build_dt_dataset.py.")]
    public float lookAheadDistance = 1.5f;
    [Tooltip("Jak daleko do przodu szukac rzutu auta na trase przy kazdej klatce.")]
    public float searchForward = 6f;
    [Tooltip("Ile wstecz. Male, zeby auto nie zrzutowalo sie na wczesniejszy fragment trasy w rownoleglym korytarzu.")]
    public float searchBackward = 1f;
    public float searchStep = 0.2f;
    [Tooltip("O ile metrow ponad faktycznie przejechany dystans postep moze wzrosnac w jednej klatce. Bez tego ograniczenia rzut przeskakuje na rownolegly fragment trasy (np. tor powrotny odnogi) i cala jej dlugosc jest uznawana za przejechana bez jezdzenia.")]
    public float progressSlack = 0.3f;
    [Tooltip("Powyzej tego odchylenia od trasy szukamy rzutu po CALEJ trasie, nie tylko w oknie.")]
    public float maxDeviation = 4f;

    [Header("Tryb pracy")]
    [Tooltip("Gdy false, AutoExplorer LICZY expertLocalWaypoint, ale NIE rusza obiektu target. "
           + "Uzywane przez CoverageBenchmark (polityka ExpertFrozen), zeby ekspert dzialal przez "
           + "TEN SAM interfejs co DT - jeden zamrozony waypoint na decyzje zamiast przeliczania "
           + "co klatke. Bez tego porownanie DT vs AutoExplorer mierzy warstwe aktuacji, a nie "
           + "polityke. Przydatne rowniez jako nauczyciel w tle przy DAggerze.")]
    public bool driveTarget = true;

    [Header("Spawn")]
    [Tooltip("Start w losowym punkcie trasy - daje zroznicowane pozycje poczatkowe, ktorych ciagle nagrywanie nie dawalo.")]
    public bool respawnOnStart = true;
    [Tooltip("Losowe odchylenie od kierunku trasy przy spawnie, w stopniach (+/-). 180 = pelna losowosc. Przy pelnej losowosci co drugi respawn zaczyna sie od zawracania, ktore detektor bierze za zaklinowanie - stad domyslne 90.")]
    public float spawnYawJitter = 90f;
    public float spawnHeightOffset = 0.2f;

    [Header("Utkniecie")]
    public bool detectStuck = true;
    [Tooltip("Przez ile sekund auto musi nie ruszyc sie o stuckDistanceThreshold, zeby uznac je za zaklinowane. Za male wartosci lapia normalne manewry zawracania.")]
    public float stuckCheckWindow = 6f;
    public float stuckDistanceThreshold = 0.5f;
    [Tooltip("Karencja po respawnie - przez tyle sekund nie sprawdzamy zaklinowania. Auto ustawione bokiem do trasy potrzebuje czasu na manewr, a bez karencji zostaloby natychmiast uznane za zaklinowane i przeniesione ponownie.")]
    public float spawnGracePeriod = 5f;
    [Tooltip("Po wykryciu zaklinowania: zakoncz epizod w loggerze (jesli podpiety) i przenies auto w losowy punkt trasy. Bezpieczne TYLKO z podpietym dataLogger - inaczej powstaje teleportacja w srodku trajektorii.")]
    public bool respawnWhenStuck = true;

    [Header("Runtime (read-only)")]
    public bool isExploring = false;
    public float progressAlongRoute = 0f;
    public float deviationFromRoute = 0f;
    public int lapsCompleted = 0;
    public int stuckEvents = 0;
    [Tooltip("Postep na trasie przy kolejnych zaklinowaniach. Skupienie wartosci = konkretne zle miejsce na trasie. Rozrzut = problem z driverem.")]
    public string stuckHotspots = "";
    [Tooltip("ETYKIETA dla DT: wektor do pursuit pointa w ukladzie auta. x = w prawo, z = do przodu, w metrach.")]
    public Vector2 expertLocalWaypoint;
    [Tooltip("Pursuit point w ukladzie SWIATA. Liczony zawsze, niezaleznie od driveTarget - "
           + "wlasciwosc ExpertWorldWaypoint nie moze czytac target.position, bo przy "
           + "driveTarget = false celem steruje kto inny (DT albo benchmark).")]
    public Vector3 expertWorldWaypointRaw;

    private float stuckTimer = 0f;
    private float graceTimer = 0f;
    private Vector3 stuckAnchor;
    private float routeLength = 0f;
    private Vector3 lastProjectionPos;
    private readonly System.Collections.Generic.List<float> stuckAt =
        new System.Collections.Generic.List<float>();

    public Vector3 ExpertWorldWaypoint => expertWorldWaypointRaw;
    public bool StuckThisFrame { get; private set; }

    // =======================================================================

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
        isExploring = true;

        UpdateTarget();
        Debug.Log($"[AutoExplorer] Start. Trasa {routeLength:F1} m, "
                + $"postep {progressAlongRoute:F1} m");
    }

    public void StopExploring() => isExploring = false;

    /// <summary>Przenosi auto na losowy punkt trasy. Daje zroznicowane pozycje
    /// startowe - przy ciaglym nagrywaniu wszystkie epizody zaczynaly sie tam,
    /// gdzie skonczyl sie poprzedni.</summary>
    public void RespawnOnRoute()
    {
        float d = Random.Range(0f, routeLength);
        Vector3 pos = route.PointAtDistance(d);

        // kierunek trasy w tym punkcie - jako baza dla orientacji
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

        progressAlongRoute = d;
        stuckAnchor = carTransform.position;
        lastProjectionPos = carTransform.position;   // respawn omija limit przyrostu
        stuckTimer = 0f;
        graceTimer = spawnGracePeriod;
    }

    void Update()
    {
        if (!isExploring) return;

        UpdateProgress();
        CheckStuck();
        UpdateTarget();
    }

    // =======================================================================

    /// <summary>Rzut auta na trase w oknie wokol dotychczasowego postepu.
    /// Okno wsteczne jest male celowo: bez tego auto w korytarzu biegnacym
    /// rownolegle do wczesniejszego fragmentu trasy zrzutowaloby sie na tamten
    /// fragment i pursuit point cofnalby sie za nie.</summary>
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

        // Zgubilismy trase - szukamy po calej dlugosci. To swiadome odzyskanie,
        // wiec omija limit przyrostu.
        if (deviationFromRoute > maxDeviation)
        {
            bestD = ProjectGlobally();
            deviationFromRoute = (Flat(route.PointAtDistance(bestD)) - car).magnitude;
            lastProjectionPos = carTransform.position;
        }
        else
        {
            // Postep nie moze wzrosnac bardziej, niz auto faktycznie przejechalo.
            // Bez tego rzut przeskakuje na rownolegly fragment trasy - tak
            // "znikala" cala odnoga przy searchForward = 6.
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

    private void UpdateTarget()
    {
        Vector3 wp = route.PointAtDistance(progressAlongRoute + lookAheadDistance);

        // ETYKIETA liczy sie ZAWSZE, takze gdy ekspert nie steruje autem.
        // Dzieki temu DTDataLogger dziala bez zmian w kazdym trybie, a przy
        // DAggerze ekspert moze byc nauczycielem w tle.
        Vector3 local = Quaternion.Inverse(
            Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f)) * Flat(wp - carTransform.position);
        expertLocalWaypoint = new Vector2(local.x, local.z);
        expertWorldWaypointRaw = wp;

        if (driveTarget)
            target.position = wp + new Vector3(0, 0.15f, 0);
    }

    private void CheckStuck()
    {
        StuckThisFrame = false;
        if (!detectStuck) return;

        // Karencja: tuz po respawnie auto czesto stoi bokiem do trasy i manewruje.
        // Bez tego kazdy taki manewr konczylby sie kolejnym respawnem i kolejnym
        // odrzuconym fragmentem - stad 28 odrzuconych na 13 zapisanych.
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
                // przenosimy auto. Odwrotnie teleportacja trafilaby do buforu
                // i relabeling zobaczylby skok o kilkanascie metrow.
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

        // histogram co 5 m trasy - pokazuje, czy zaklinowania sa skupione
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

        Gizmos.color = Color.green;                       // pursuit point EKSPERTA
        Gizmos.DrawWireSphere(expertWorldWaypointRaw + Vector3.up * 0.1f, 0.3f);
        Gizmos.DrawLine(carTransform.position, expertWorldWaypointRaw);

        if (!driveTarget && target != null)                // aktywny cel kogos innego
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