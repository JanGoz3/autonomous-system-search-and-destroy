using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.InputSystem;

/// <summary>
/// Mierzy to, o co w projekcie naprawde chodzi: ile metrow kwadratowych pietra
/// auto pokrywa w zadanym czasie.
///
/// Wzgledem poprzedniej wersji:
///  - czwarta polityka ExpertFrozen: AutoExplorer przez TEN SAM interfejs co DT
///    (jeden zamrozony waypoint co decisionInterval). Bez tego porownanie DT vs
///    AutoExplorer mierzy warstwe aktuacji (co klatka vs co 1.5 s), nie polityke.
///  - forceFairExpert: wylacza respawnOnStart / respawnWhenStuck na czas
///    przebiegu. Respawn na starcie ignoruje sparowana pozycje startowa, a
///    respawn przy zaklinowaniu wlicza teleportacje do pokrycia i do dystansu.
///  - dump konfiguracji (trainingMode, oba respawny) do logu.
///  - kontrola, czy pozycje startowe mieszcza sie w obszarze danych treningowych.
///  - liczniki per polityka w CSV (poza NavMesh, cel za autem, decyzje w bezruchu).
/// </summary>
public class CoverageBenchmark : MonoBehaviour
{
    public enum Policy { DecisionTransformer, ExpertFrozen, RandomWaypoint, AutoExplorer }

    [Header("References")]
    public Transform carTransform;
    public Rigidbody carRigidbody;
    public Transform target;
    public CarAgent carAgent;
    public DTInference dtInference;
    public AutoExplorer autoExplorer;

    [Header("Protokol")]
    [Tooltip("Dlugosc jednego przebiegu w sekundach czasu symulacji.")]
    public float runSeconds = 300f;
    [Tooltip("Ile przebiegow na polityke. Kazdy numer przebiegu ma ten sam punkt startowy dla wszystkich polityk.")]
    public int runsPerPolicy = 5;
    public int seed = 12345;
    [Tooltip("Przyspieszenie symulacji. 4x zwykle jest bezpieczne; wyzej fizyka moze sie psuc.")]
    public float timeScale = 4f;

    [Header("Uczciwosc porownania")]
    [Tooltip("Na czas przebiegu wylacza AutoExplorer.respawnOnStart i respawnWhenStuck, po czym przywraca poprzednie wartosci. Bez tego ekspert startuje z losowego punktu trasy (nie ze sparowanej pozycji) i teleportuje sie przy zaklinowaniu, co wchodzi i do pokrycia, i do dystansu.")]
    public bool forceFairExpert = true;
    [Tooltip("Ostrzega, gdy wylosowana pozycja startowa lezy poza obszarem, z ktorego pochodza dane treningowe. Model uczyl sie etykiety bedacej praktycznie funkcja (posX, posZ, yaw) - poza tym prostokatem ekstrapoluje.")]
    public bool warnOutsideTrainingArea = true;
    public Vector2 trainAreaMin = new Vector2(-2.5f, 7.2f);
    public Vector2 trainAreaMax = new Vector2(15.4f, 27.4f);

    [Header("Metryka")]
    public float gridCellSize = 1.0f;
    [Tooltip("Co ile sekund zapisac punkt krzywej pokrycia.")]
    public float sampleInterval = 5f;

    [Header("Baseline: losowy waypoint")]
    [Tooltip("Ta sama odleglosc co WAYPOINT_DIST w build_dt_dataset.py.")]
    public float randomWaypointDistance = 1.5f;
    [Tooltip("Ten sam interwal co decisionInterval w DTInference. Uzywany takze przez ExpertFrozen i RandomWaypoint.")]
    public float decisionInterval = 1.5f;

    [Header("Co uruchomic")]
    [Tooltip("Puste = wszystkie cztery polityki. Wpisz np. tylko ExpertFrozen, zeby zmierzyc sam sufit interfejsu.")]
    public List<Policy> policiesToRun = new List<Policy>();
    [Tooltip("Puste = uzyj decisionInterval powyzej. Wpisz kilka wartosci (np. 0.25, 0.5, 1, 1.5), zeby przemiesc interwal decyzji. Kazda wartosc to osobny wiersz w wynikach, opisany jako 'Polityka@interwal'. Nie dotyczy AutoExplorera, ktory z definicji przelicza cel co klatke.")]
    public List<float> sweepDecisionIntervals = new List<float>();

    [Header("Spawn")]
    [Tooltip("Losuje pozycje startowe WZDLUZ TRASY zamiast w prostokacie areny. Trasa jest z definicji przejezdna, a NavMesh.SamplePosition przyjmuje punkty przy scianach i w narozdnikach - poprawne dla agenta o promieniu, ale nie dla pojazdu o promieniu skretu. Wymaga podpietego AutoExplorer z referencja do CoverageRoute.")]
    public bool spawnOnRoute = true;
    [Tooltip("Losowe odchylenie orientacji od kierunku trasy, w stopniach (+/-). Zero testowaloby wylacznie najlatwiejszy przypadek; odchylenie jest tym, czego model i tak musi sie nauczyc.")]
    public float spawnYawJitter = 90f;
    [Tooltip("Losowe przesuniecie prostopadle do trasy, w metrach (+/-). Zapobiega temu, zeby wszystkie starty lezaly dokladnie na polilinii.")]
    public float spawnLateralOffset = 0.3f;
    public float spawnHeightOffset = 0.2f;
    [Tooltip("Odrzuca pozycje, w ktorych auto ma blizej niz tyle metrow do przeszkody w ktoryms z czterech kierunkow. 0 = bez sprawdzania.")]
    public float minClearance = 0.5f;

    [Tooltip("Uzywane TYLKO gdy spawnOnRoute = false. Domyslne +/-20 m to obszar znacznie wiekszy niz pietro.")]
    public Vector2 arenaMin = new Vector2(-20f, -20f);
    public Vector2 arenaMax = new Vector2(20f, 20f);

    [Header("Kolizje (opcjonalne)")]
    [Tooltip("Wymaga pola 'public int collisionCount' w CarAgent.cs. hadCollisionThisStep sie NIE nadaje - nigdzie nie jest ustawiane na true, a dodatkowo DTInference i DTDataLogger je konsumuja.")]
    public bool countCollisions = false;

    [Header("Wyjscie")]
    public string outputFolder = "DTBenchmark";

    [Header("Runtime (read-only)")]
    public bool running = false;
    public string status = "nacisnij B";

    private readonly StringBuilder csv = new StringBuilder();
    private readonly StringBuilder summary = new StringBuilder();

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null && kb.bKey.wasPressedThisFrame && !running)
            StartCoroutine(RunBenchmark());
    }

    private IEnumerator RunBenchmark()
    {
        running = true;
        csv.Clear();
        summary.Clear();
        csv.AppendLine("policy,run,t,cells,area_m2,distance_m,collisions");
        summary.AppendLine("policy,run,cells,area_m2,distance_m,decisions," +
                           "off_navmesh,behind,stalled,stuck_events,laps,start_x,start_z,start_in_train_area");

        DumpConfig();

        // Pozycje startowe losowane RAZ i uzywane przez WSZYSTKIE polityki.
        var rng = new System.Random(seed);
        var starts = BuildStarts(rng);
        if (starts.Count < runsPerPolicy)
        {
            Debug.LogError($"[Benchmark] Udalo sie wylosowac tylko {starts.Count}/{runsPerPolicy} " +
                           $"pozycji startowych. Zmniejsz minClearance albo sprawdz referencje do trasy.");
            running = false;
            yield break;
        }

        int outside = 0;
        for (int i = 0; i < starts.Count; i++)
        {
            bool inArea = InTrainArea(starts[i].position);
            if (!inArea) outside++;
            Debug.Log($"[Benchmark] start {i}: X={starts[i].position.x:F1} " +
                      $"Z={starts[i].position.z:F1}  " +
                      $"{(inArea ? "w obszarze danych" : "POZA OBSZAREM DANYCH")}");
        }
        if (warnOutsideTrainingArea && outside > 0)
            Debug.LogWarning($"[Benchmark] {outside}/{starts.Count} pozycji startowych lezy poza " +
                             $"X[{trainAreaMin.x};{trainAreaMax.x}] Z[{trainAreaMin.y};{trainAreaMax.y}]. " +
                             $"Etykieta DT jest praktycznie funkcja (posX, posZ, yaw) - tam model ekstrapoluje. " +
                             $"Rozwaz zawezenie arenaMin/arenaMax do obszaru trasy.");

        // zapamietujemy i wylaczamy respawny eksperta
        bool prevRespawnStart = false, prevRespawnStuck = false, prevDrive = true;
        if (autoExplorer != null)
        {
            prevRespawnStart = autoExplorer.respawnOnStart;
            prevRespawnStuck = autoExplorer.respawnWhenStuck;
            prevDrive = autoExplorer.driveTarget;
            if (forceFairExpert)
            {
                autoExplorer.respawnOnStart = false;
                autoExplorer.respawnWhenStuck = false;
                Debug.Log("[Benchmark] forceFairExpert: respawnOnStart i respawnWhenStuck " +
                          "wylaczone na czas przebiegu.");
            }
        }

        Time.timeScale = timeScale;

        // lista zadan: kazda polityka x kazdy interwal do przemiecenia
        var policies = (policiesToRun != null && policiesToRun.Count > 0)
            ? new List<Policy>(policiesToRun)
            : new List<Policy>((Policy[])System.Enum.GetValues(typeof(Policy)));
        var intervals = (sweepDecisionIntervals != null && sweepDecisionIntervals.Count > 0)
            ? new List<float>(sweepDecisionIntervals)
            : new List<float> { decisionInterval };

        var jobs = new List<(Policy pol, float iv, string label)>();
        foreach (Policy p in policies)
        {
            // AutoExplorer przelicza cel co klatke - interwal go nie dotyczy
            if (p == Policy.AutoExplorer || intervals.Count == 1)
                jobs.Add((p, intervals[0], p.ToString()));
            else
                foreach (float iv in intervals)
                    jobs.Add((p, iv,
                        $"{p}@{iv.ToString("0.##", CultureInfo.InvariantCulture)}"));
        }
        Debug.Log($"[Benchmark] Do wykonania: {jobs.Count} x {runsPerPolicy} przebiegow " +
                  $"({string.Join(", ", jobs.ConvertAll(j => j.label))})");

        foreach (var job in jobs)
            for (int run = 0; run < runsPerPolicy; run++)
            {
                status = $"{job.label} przebieg {run + 1}/{runsPerPolicy}";
                yield return StartCoroutine(RunOne(job.pol, run, starts[run], job.label, job.iv));
            }

        Time.timeScale = 1f;

        if (dtInference != null) dtInference.decisionInterval = decisionInterval;
        if (autoExplorer != null)
        {
            autoExplorer.respawnOnStart = prevRespawnStart;
            autoExplorer.respawnWhenStuck = prevRespawnStuck;
            autoExplorer.driveTarget = prevDrive;
        }

        SaveCsv();
        running = false;
        status = "gotowe";
    }

    private List<Pose> BuildStarts(System.Random rng)
    {
        var starts = new List<Pose>();

        CoverageRoute route = autoExplorer != null ? autoExplorer.route : null;
        bool useRoute = spawnOnRoute && route != null && route.TotalLength() > 1f;

        if (spawnOnRoute && !useRoute)
            Debug.LogWarning("[Benchmark] spawnOnRoute = true, ale brak trasy " +
                             "(autoExplorer.route). Wracam do losowania w prostokacie areny.");

        float len = useRoute ? route.TotalLength() : 0f;
        int guard = 0;

        while (starts.Count < runsPerPolicy && guard++ < 20000)
        {
            Vector3 pos;
            Quaternion rot;

            if (useRoute)
            {
                float d = (float)rng.NextDouble() * len;
                pos = route.PointAtDistance(d);

                Vector3 ahead = route.PointAtDistance(d + 1f);
                Vector3 dir = ahead - pos;
                dir.y = 0f;
                if (dir.sqrMagnitude < 1e-4f) dir = Vector3.forward;
                rot = Quaternion.LookRotation(dir.normalized, Vector3.up);

                if (spawnLateralOffset > 0f)
                {
                    Vector3 side = Vector3.Cross(Vector3.up, dir.normalized);
                    pos += side * (float)((rng.NextDouble() * 2.0 - 1.0) * spawnLateralOffset);
                }
                if (spawnYawJitter > 0f)
                    rot *= Quaternion.Euler(0f,
                        (float)((rng.NextDouble() * 2.0 - 1.0) * spawnYawJitter), 0f);
            }
            else
            {
                pos = new Vector3(
                    Mathf.Lerp(arenaMin.x, arenaMax.x, (float)rng.NextDouble()), 0f,
                    Mathf.Lerp(arenaMin.y, arenaMax.y, (float)rng.NextDouble()));
                rot = Quaternion.Euler(0f, (float)rng.NextDouble() * 360f, 0f);
            }

            if (!NavMesh.SamplePosition(pos, out var hit, 2f, NavMesh.AllAreas))
                continue;

            Vector3 final = hit.position + Vector3.up * spawnHeightOffset;
            if (!HasClearance(final)) continue;

            starts.Add(new Pose(final, rot));
        }

        return starts;
    }

    private bool HasClearance(Vector3 p)
    {
        if (minClearance <= 0f) return true;
        Vector3[] dirs = { Vector3.forward, Vector3.back, Vector3.left, Vector3.right };
        foreach (var d in dirs)
            if (Physics.Raycast(p, d, minClearance))
                return false;
        return true;
    }

    private void DumpConfig()
    {
        string tm = carAgent == null ? "brak referencji" : carAgent.trainingMode.ToString();
        string ros = autoExplorer == null ? "brak" : autoExplorer.respawnOnStart.ToString();
        string rws = autoExplorer == null ? "brak" : autoExplorer.respawnWhenStuck.ToString();
        Debug.Log($"[Benchmark] KONFIGURACJA:\n" +
                  $"  CarAgent.trainingMode         = {tm}\n" +
                  $"  AutoExplorer.respawnOnStart   = {ros}\n" +
                  $"  AutoExplorer.respawnWhenStuck = {rws}\n" +
                  $"  forceFairExpert               = {forceFairExpert}\n" +
                  $"  spawnOnRoute                  = {spawnOnRoute}" +
                  (spawnOnRoute
                      ? $"  (jitter +/-{spawnYawJitter} st, offset +/-{spawnLateralOffset} m, " +
                        $"clearance {minClearance} m)\n"
                      : $"  -> arena X[{arenaMin.x};{arenaMax.x}] Z[{arenaMin.y};{arenaMax.y}]\n") +
                  $"  runSeconds={runSeconds}  runsPerPolicy={runsPerPolicy}  " +
                  $"decisionInterval={decisionInterval.ToString("0.##", CultureInfo.InvariantCulture)}");

        if (carAgent != null && carAgent.trainingMode)
            Debug.LogError("[Benchmark] CarAgent.trainingMode = TRUE. Dojechanie do waypointa " +
                           "wywola EndEpisode(), ktore teleportuje auto i NADPISZE Target.position " +
                           "wlasnym spawnem. Wyniki beda bezwartosciowe. Odznacz przed pomiarem.");
        if (dtInference != null && dtInference.initialTargetReturn < 10f)
            Debug.LogWarning($"[Benchmark] DTInference.initialTargetReturn = " +
                             $"{dtInference.initialTargetReturn}. Przy etykietach eksperta sumy nagrod " +
                             $"sa dodatnie (rzedu 33..118), a RTG bliskie zera wystepuje tylko na " +
                             $"KONCU epizodu - model jest warunkowany na zachowanie terminalne.");
    }

    private bool InTrainArea(Vector3 p) =>
        p.x >= trainAreaMin.x && p.x <= trainAreaMax.x &&
        p.z >= trainAreaMin.y && p.z <= trainAreaMax.y;

    private IEnumerator RunOne(Policy policy, int run, Pose start, string label, float interval)
    {
        StopAll();

        carTransform.SetPositionAndRotation(start.position, start.rotation);
        if (carRigidbody != null)
        {
            carRigidbody.linearVelocity = Vector3.zero;
            carRigidbody.angularVelocity = Vector3.zero;
        }
        if (target != null) target.position = start.position;
        yield return new WaitForFixedUpdate();

        int collisionsAtStart = ReadCollisions();
        var visited = new HashSet<Vector2Int>();
        float distance = 0f;
        Vector3 lastPos = carTransform.position;

        switch (policy)
        {
            case Policy.DecisionTransformer:
                if (dtInference != null)
                {
                    dtInference.decisionLogTag = $"{label}_run{run}";
                    dtInference.decisionInterval = interval;
                }
                dtInference.StartInference();
                break;

            case Policy.ExpertFrozen:
                // ekspert liczy waypoint, ale celem steruje benchmark - raz na decisionInterval
                autoExplorer.driveTarget = false;
                autoExplorer.StartExploring();
                break;

            case Policy.AutoExplorer:
                autoExplorer.driveTarget = true;
                autoExplorer.StartExploring();
                break;
        }

        float t = 0f, nextSample = 0f, nextWaypoint = 0f;
        while (t < runSeconds)
        {
            if (policy == Policy.RandomWaypoint && t >= nextWaypoint)
            {
                nextWaypoint = t + interval;
                float a = Random.Range(-Mathf.PI, Mathf.PI);            // pelny okrag,
                float yaw = carTransform.eulerAngles.y * Mathf.Deg2Rad;  // tak jak 36 binow DT
                float lx = Mathf.Sin(a) * randomWaypointDistance;
                float lz = Mathf.Cos(a) * randomWaypointDistance;
                float c = Mathf.Cos(yaw), s = Mathf.Sin(yaw);
                target.position = carTransform.position
                                + new Vector3(lx * c + lz * s, 0f, -lx * s + lz * c);
            }

            if (policy == Policy.ExpertFrozen && t >= nextWaypoint)
            {
                nextWaypoint = t + interval;
                target.position = autoExplorer.ExpertWorldWaypoint + new Vector3(0, 0.05f, 0);
            }

            Vector3 pos = carTransform.position;
            distance += Vector3.Distance(new Vector3(pos.x, 0, pos.z),
                                         new Vector3(lastPos.x, 0, lastPos.z));
            lastPos = pos;
            visited.Add(new Vector2Int(Mathf.FloorToInt(pos.x / gridCellSize),
                                       Mathf.FloorToInt(pos.z / gridCellSize)));

            if (t >= nextSample)
            {
                nextSample += sampleInterval;
                int col = ReadCollisions() - collisionsAtStart;
                var ci = CultureInfo.InvariantCulture;
                csv.AppendLine($"{label},{run},{t.ToString("F1", ci)},{visited.Count}," +
                               $"{(visited.Count * gridCellSize * gridCellSize).ToString("F1", ci)}," +
                               $"{distance.ToString("F2", ci)},{col}");
            }

            yield return null;
            t += Time.deltaTime;
        }

        // liczniki zbierane PRZED StopAll, bo StopInference je wypisuje i zapisuje log
        int decisions = 0, offMesh = 0, behind = 0, stalled = 0, stuckEv = 0, laps = 0;
        if (policy == Policy.DecisionTransformer && dtInference != null)
        {
            decisions = dtInference.decisionCount;
            offMesh = dtInference.waypointsOffNavMesh;
            behind = dtInference.decisionsBehind;
            stalled = dtInference.decisionsWhileStalled;
        }
        if ((policy == Policy.AutoExplorer || policy == Policy.ExpertFrozen) && autoExplorer != null)
        {
            stuckEv = autoExplorer.stuckEvents;
            laps = autoExplorer.lapsCompleted;
        }

        StopAll();

        var inv = CultureInfo.InvariantCulture;
        summary.AppendLine($"{label},{run},{visited.Count}," +
            $"{(visited.Count * gridCellSize * gridCellSize).ToString("F1", inv)}," +
            $"{distance.ToString("F2", inv)},{decisions},{offMesh},{behind},{stalled}," +
            $"{stuckEv},{laps},{start.position.x.ToString("F2", inv)}," +
            $"{start.position.z.ToString("F2", inv)},{(InTrainArea(start.position) ? 1 : 0)}");

        Debug.Log($"[Benchmark] {label} run {run}: {visited.Count} komorek, {distance:F1} m" +
                  (policy == Policy.DecisionTransformer
                      ? $", decyzji={decisions}, pozaNavMesh={offMesh}, " +
                        $"celZaAutem={behind}, wBezruchu={stalled}"
                      : "") +
                  ((policy == Policy.AutoExplorer || policy == Policy.ExpertFrozen)
                      ? $", zaklinowan={stuckEv}, okrazen={laps}" : ""));
    }

    /// <summary>Zwraca 0, gdy zliczanie kolizji jest wylaczone.</summary>
    private int ReadCollisions()
    {
        if (!countCollisions || carAgent == null) return 0;
        // return carAgent.collisionCount;   // <- odkomentuj po dodaniu pola w CarAgent
        return 0;
    }

    private void StopAll()
    {
        if (dtInference != null && dtInference.isActive) dtInference.StopInference();
        if (autoExplorer != null) autoExplorer.StopExploring();
    }

    private void SaveCsv()
    {
        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        Directory.CreateDirectory(dir);
        string stamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");

        string p1 = Path.Combine(dir, $"benchmark_{stamp}.csv");
        File.WriteAllText(p1, csv.ToString());

        string p2 = Path.Combine(dir, $"summary_{stamp}.csv");
        File.WriteAllText(p2, summary.ToString());

        Debug.Log($"[Benchmark] Zapisano:\n  {p1}\n  {p2}");
    }
}