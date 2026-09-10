using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.InputSystem;

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
    [Tooltip("Opcjonalne. Czyszczone po teleportacji, przed okresem rozgrzewki.")]
    public TofScanBuffer tofScanBuffer;

    [Header("Protokol")]
    [Tooltip("Dlugosc jednego przebiegu w sekundach czasu symulacji.")]
    public float runSeconds = 180f;
    [Tooltip("Ile przebiegow na polityke. Kazdy numer przebiegu ma ten sam punkt startowy "
           + "dla wszystkich polityk.")]
    public int runsPerPolicy = 5;
    public int seed = 12345;
    [Tooltip("Przyspieszenie symulacji. Przy przemiataniu interwalu decyzji trzymaj 1-2: "
           + "przy wyzszych wartosciach dlugosc klatki zaczyna byc porownywalna z interwalem "
           + "i faktyczny interwal rozjezdza sie z zadanym.")]
    public float timeScale = 2f;
    [Tooltip("Ile sekund auto stoi po teleportacji, zanim ruszy polityka. Daje wiezyczce czas "
           + "na wypelnienie profilu ToF - inaczej pierwsza decyzja zapada na pustym buforze.")]
    public float warmupSeconds = 1.5f;

    [Header("Uczciwosc porownania")]
    [Tooltip("Na czas przebiegu wylacza AutoExplorer.respawnOnStart i respawnWhenStuck, "
           + "po czym przywraca poprzednie wartosci.")]
    public bool forceFairExpert = true;
    [Tooltip("Ostrzega, gdy wylosowana pozycja startowa lezy poza obszarem, z ktorego "
           + "pochodza dane treningowe.")]
    public bool warnOutsideTrainingArea = true;
    public Vector2 trainAreaMin = new Vector2(-2.5f, 7.2f);
    public Vector2 trainAreaMax = new Vector2(15.4f, 27.4f);

    [Header("Metryka")]
    public float gridCellSize = 1.0f;
    [Tooltip("Co ile sekund zapisac punkt krzywej pokrycia.")]
    public float sampleInterval = 5f;
    [Tooltip("Ponizej tego przesuniecia miedzy taktami decyzji auto uznajemy za stojace.")]
    public float stallThreshold = 0.1f;

    [Header("Baseline: losowy waypoint")]
    [Tooltip("Ta sama odleglosc co WAYPOINT_DIST w build_dt_dataset.py.")]
    public float randomWaypointDistance = 1.5f;
    [Tooltip("Ten sam interwal co decisionInterval w DTInference. Uzywany takze przez "
           + "ExpertFrozen i RandomWaypoint.")]
    public float decisionInterval = 1.5f;

    [Header("Co uruchomic")]
    [Tooltip("Puste = wszystkie cztery polityki.")]
    public List<Policy> policiesToRun = new List<Policy>();
    [Tooltip("Puste = uzyj decisionInterval powyzej. Wpisz kilka wartosci (np. 0.5, 1, 1.5), "
           + "zeby przemiesc interwal decyzji. Nie dotyczy AutoExplorera, ktory z definicji "
           + "przelicza cel co klatke.")]
    public List<float> sweepDecisionIntervals = new List<float>();

    [Header("Spawn")]
    [Tooltip("Losuje pozycje startowe WZDLUZ TRASY zamiast w prostokacie areny.")]
    public bool spawnOnRoute = true;
    [Tooltip("Losowe odchylenie orientacji od kierunku trasy, w stopniach (+/-).")]
    public float spawnYawJitter = 90f;
    [Tooltip("Losowe przesuniecie prostopadle do trasy, w metrach (+/-).")]
    public float spawnLateralOffset = 0.3f;
    public float spawnHeightOffset = 0.2f;
    [Tooltip("Odrzuca pozycje, w ktorych auto ma blizej niz tyle metrow do przeszkody "
           + "w ktoryms z czterech kierunkow. 0 = bez sprawdzania.")]
    public float minClearance = 0.5f;

    [Tooltip("Uzywane TYLKO gdy spawnOnRoute = false.")]
    public Vector2 arenaMin = new Vector2(-20f, -20f);
    public Vector2 arenaMax = new Vector2(20f, 20f);

    [Header("Kolizje (opcjonalne)")]
    [Tooltip("Wymaga pola 'public int collisionCount' w CarAgent.cs.")]
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
        summary.AppendLine("policy,run,interval,cells,area_m2,distance_m,ticks," +
                           "behind_pct,stalled_pct,mean_abs_angle,dt_decisions,dt_off_navmesh," +
                           "dt_behind,dt_stalled,stuck_events,laps,start_x,start_z,start_in_train_area");

        DumpConfig();

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
                             $"Tam model ekstrapoluje.");

        bool prevRespawnStart = false, prevRespawnStuck = false, prevDrive = true;
        float prevDtInterval = decisionInterval;
        if (autoExplorer != null)
        {
            prevRespawnStart = autoExplorer.respawnOnStart;
            prevRespawnStuck = autoExplorer.respawnWhenStuck;
            prevDrive = autoExplorer.driveTarget;
            if (forceFairExpert)
            {
                autoExplorer.respawnOnStart = false;
                autoExplorer.respawnWhenStuck = false;
                Debug.Log("[Benchmark] forceFairExpert: respawnOnStart i respawnWhenStuck wylaczone.");
            }
        }
        if (dtInference != null) prevDtInterval = dtInference.decisionInterval;

        Time.timeScale = timeScale;

        var policies = (policiesToRun != null && policiesToRun.Count > 0)
            ? new List<Policy>(policiesToRun)
            : new List<Policy>((Policy[])System.Enum.GetValues(typeof(Policy)));
        var intervals = (sweepDecisionIntervals != null && sweepDecisionIntervals.Count > 0)
            ? new List<float>(sweepDecisionIntervals)
            : new List<float> { decisionInterval };

        var jobs = new List<(Policy pol, float iv, string label)>();
        foreach (Policy p in policies)
        {
            if (p == Policy.AutoExplorer || intervals.Count == 1)
                jobs.Add((p, intervals[0], p.ToString()));
            else
                foreach (float iv in intervals)
                    jobs.Add((p, iv,
                        $"{p}@{iv.ToString("0.##", CultureInfo.InvariantCulture)}"));
        }
        Debug.Log($"[Benchmark] Do wykonania: {jobs.Count} x {runsPerPolicy} przebiegow " +
                  $"({string.Join(", ", jobs.ConvertAll(j => j.label))}). " +
                  $"Szacowany czas: {jobs.Count * runsPerPolicy * (runSeconds + warmupSeconds) / Mathf.Max(timeScale, 0.01f) / 60f:F0} min.");

        foreach (var job in jobs)
            for (int run = 0; run < runsPerPolicy; run++)
            {
                status = $"{job.label} przebieg {run + 1}/{runsPerPolicy}";
                yield return StartCoroutine(RunOne(job.pol, run, starts[run], job.label, job.iv));
            }

        Time.timeScale = 1f;

        if (dtInference != null) dtInference.decisionInterval = prevDtInterval;
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
            Debug.LogWarning("[Benchmark] spawnOnRoute = true, ale brak trasy. "
                           + "Wracam do losowania w prostokacie areny.");

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
                  $"warmup={warmupSeconds}s  timeScale={timeScale}");

        if (carAgent != null && carAgent.trainingMode)
            Debug.LogError("[Benchmark] CarAgent.trainingMode = TRUE. Dojechanie do waypointa "
                         + "wywola EndEpisode(), ktore teleportuje auto i NADPISZE Target.position. "
                         + "Wyniki beda bezwartosciowe. Odznacz przed pomiarem.");

        bool sweeping = sweepDecisionIntervals != null && sweepDecisionIntervals.Count > 1;
        if (sweeping && timeScale > 2f)
            Debug.LogWarning($"[Benchmark] timeScale={timeScale} przy przemiataniu interwalu. "
                           + "Dlugosc klatki w czasie symulacji rosnie proporcjonalnie do timeScale, "
                           + "wiec faktyczny interwal decyzji kwantuje sie do wielokrotnosci klatki. "
                           + "Zjedz do 1-2, inaczej mierzysz zaszumiona wersje zmiennej, ktora badasz.");

        if (dtInference != null && sweeping)
        {
            float minIv = float.MaxValue;
            foreach (float iv in sweepDecisionIntervals) minIv = Mathf.Min(minIv, iv);
            int maxDecisions = Mathf.CeilToInt(runSeconds / Mathf.Max(minIv, 0.01f));
            if (dtInference.pseudoEpisodeDecisions <= 0 && maxDecisions > dtInference.maxEpLen)
                Debug.LogWarning($"[Benchmark] Przy interwale {minIv}s przebieg ma ~{maxDecisions} "
                               + $"decyzji, a maxEpLen={dtInference.maxEpLen}. Timesteps beda "
                               + "przyciete przez wiekszosc przebiegu. Rozwaz ustawienie "
                               + "DTInference.pseudoEpisodeDecisions (np. 53).");
        }
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

        if (tofScanBuffer != null) tofScanBuffer.Clear();
        yield return new WaitForFixedUpdate();

        if (warmupSeconds > 0f) yield return new WaitForSeconds(warmupSeconds);

        int collisionsAtStart = ReadCollisions();
        var visited = new HashSet<Vector2Int>();
        float distance = 0f;
        Vector3 lastPos = carTransform.position;

        switch (policy)
        {
            case Policy.DecisionTransformer:
                if (dtInference == null)
                {
                    Debug.LogError("[Benchmark] Brak referencji DTInference - pomijam przebieg.");
                    yield break;
                }
                dtInference.decisionLogTag = $"{label}_run{run}";
                dtInference.decisionInterval = interval;
                dtInference.StartInference(clearScanBuffer: false);   // bufor juz wypelniony
                break;

            case Policy.ExpertFrozen:
                if (autoExplorer == null) { Debug.LogError("[Benchmark] Brak AutoExplorer."); yield break; }
                autoExplorer.driveTarget = false;
                autoExplorer.StartExploring();
                break;

            case Policy.AutoExplorer:
                if (autoExplorer == null) { Debug.LogError("[Benchmark] Brak AutoExplorer."); yield break; }
                autoExplorer.driveTarget = true;
                autoExplorer.StartExploring();
                break;
        }

        int ticks = 0, tickBehind = 0, tickStalled = 0;
        float absAngleSum = 0f;
        Vector3 lastTickPos = carTransform.position;

        float t = 0f, nextSample = 0f, nextWaypoint = 0f, nextTick = 0f;

        while (t < runSeconds)
        {
            if (policy == Policy.RandomWaypoint && t >= nextWaypoint)
            {
                nextWaypoint = t + interval;
                float a = Random.Range(-Mathf.PI, Mathf.PI);
                float yaw = carTransform.eulerAngles.y * Mathf.Deg2Rad;
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

            if (t >= nextTick)
            {
                nextTick = t + interval;
                Vector3 pos0 = carTransform.position;

                Vector3 toTarget = target.position - pos0;
                toTarget.y = 0f;
                if (toTarget.sqrMagnitude > 1e-6f)
                {
                    Vector3 local = Quaternion.Inverse(
                        Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f)) * toTarget;
                    float ang = Mathf.Atan2(local.x, local.z) * Mathf.Rad2Deg;
                    absAngleSum += Mathf.Abs(ang);
                    if (Mathf.Abs(ang) > 90f) tickBehind++;
                }

                if (ticks > 0 &&
                    Vector3.Distance(new Vector3(pos0.x, 0, pos0.z),
                                     new Vector3(lastTickPos.x, 0, lastTickPos.z)) < stallThreshold)
                    tickStalled++;

                lastTickPos = pos0;
                ticks++;
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

        int dtDecisions = 0, dtOffMesh = 0, dtBehind = 0, dtStalled = 0;
        int stuckEv = 0, laps = 0;
        if (policy == Policy.DecisionTransformer && dtInference != null)
        {
            dtDecisions = dtInference.decisionCount;
            dtOffMesh = dtInference.waypointsOffNavMesh;
            dtBehind = dtInference.decisionsBehind;
            dtStalled = dtInference.decisionsWhileStalled;
        }
        if ((policy == Policy.AutoExplorer || policy == Policy.ExpertFrozen) && autoExplorer != null)
        {
            stuckEv = autoExplorer.stuckEvents;
            laps = autoExplorer.lapsCompleted;
        }

        StopAll();

        float behindPct = ticks > 0 ? 100f * tickBehind / ticks : 0f;
        float stalledPct = ticks > 1 ? 100f * tickStalled / (ticks - 1) : 0f;
        float meanAbsAngle = ticks > 0 ? absAngleSum / ticks : 0f;

        var inv = CultureInfo.InvariantCulture;
        summary.AppendLine($"{label},{run},{interval.ToString("0.##", inv)},{visited.Count}," +
            $"{(visited.Count * gridCellSize * gridCellSize).ToString("F1", inv)}," +
            $"{distance.ToString("F2", inv)},{ticks}," +
            $"{behindPct.ToString("F1", inv)},{stalledPct.ToString("F1", inv)}," +
            $"{meanAbsAngle.ToString("F1", inv)}," +
            $"{dtDecisions},{dtOffMesh},{dtBehind},{dtStalled}," +
            $"{stuckEv},{laps},{start.position.x.ToString("F2", inv)}," +
            $"{start.position.z.ToString("F2", inv)},{(InTrainArea(start.position) ? 1 : 0)}");

        Debug.Log($"[Benchmark] {label} run {run}: {visited.Count} komorek, {distance:F1} m, " +
                  $"celZaAutem={behindPct:F0}%, bezruch={stalledPct:F0}%, " +
                  $"sredni|kat|={meanAbsAngle:F0} st" +
                  (policy == Policy.DecisionTransformer
                      ? $", decyzji={dtDecisions}, pozaNavMesh={dtOffMesh}" : "") +
                  ((policy == Policy.AutoExplorer || policy == Policy.ExpertFrozen)
                      ? $", zaklinowan={stuckEv}, okrazen={laps}" : ""));
    }

    private int ReadCollisions()
    {
        if (!countCollisions || carAgent == null) return 0;
        // return carAgent.collisionCount;   // zeby dzialalo trzeba dodac pole w CarAgent
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