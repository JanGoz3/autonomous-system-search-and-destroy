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
/// auto pokrywa w zadanym czasie. Porownuje DT z dwoma punktami odniesienia
/// na IDENTYCZNYCH pozycjach startowych (sparowany test).
///
/// Zliczanie kolizji jest OPCJONALNE i domyslnie wylaczone. Zeby je wlaczyc,
/// dopisz w CarAgent.cs:
///     public int collisionCount = 0;              // pole
///     collisionCount++;                            // w OnCollisionEnter
/// a tutaj zaznacz countCollisions i podepnij carAgent.
/// (hadCollisionThisStep sie NIE nadaje, bo DTInference i DTDataLogger
///  konsumuja te flage - kto pierwszy odczyta, ten ja kasuje)
/// </summary>
public class CoverageBenchmark : MonoBehaviour
{
    public enum Policy { DecisionTransformer, RandomWaypoint, AutoExplorer }

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

    [Header("Metryka")]
    public float gridCellSize = 1.0f;
    [Tooltip("Co ile sekund zapisac punkt krzywej pokrycia.")]
    public float sampleInterval = 5f;

    [Header("Baseline: losowy waypoint")]
    [Tooltip("Ta sama odleglosc co WAYPOINT_DIST w build_dt_dataset.py.")]
    public float randomWaypointDistance = 1.5f;
    [Tooltip("Ten sam interwal co decisionInterval w DTInference.")]
    public float decisionInterval = 1.5f;

    [Header("Spawn")]
    public Vector2 arenaMin = new Vector2(-20f, -20f);
    public Vector2 arenaMax = new Vector2(20f, 20f);

    [Header("Kolizje (opcjonalne)")]
    [Tooltip("Wymaga pola 'public int collisionCount' w CarAgent.cs. Bez tego zostaw odznaczone - reszta metryk liczy sie normalnie.")]
    public bool countCollisions = false;

    [Header("Wyjscie")]
    public string outputFolder = "DTBenchmark";

    [Header("Runtime (read-only)")]
    public bool running = false;
    public string status = "nacisnij B";

    private readonly StringBuilder csv = new StringBuilder();

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
        csv.AppendLine("policy,run,t,cells,area_m2,distance_m,collisions");

        // Pozycje startowe losowane RAZ i uzywane przez wszystkie polityki.
        var starts = new List<Pose>();
        var rng = new System.Random(seed);
        while (starts.Count < runsPerPolicy)
        {
            var p = new Vector3(
                Mathf.Lerp(arenaMin.x, arenaMax.x, (float)rng.NextDouble()), 0f,
                Mathf.Lerp(arenaMin.y, arenaMax.y, (float)rng.NextDouble()));
            if (NavMesh.SamplePosition(p, out var hit, 3f, NavMesh.AllAreas))
                starts.Add(new Pose(hit.position + Vector3.up * 0.2f,
                                    Quaternion.Euler(0f, (float)rng.NextDouble() * 360f, 0f)));
        }

        Time.timeScale = timeScale;

        foreach (Policy policy in System.Enum.GetValues(typeof(Policy)))
            for (int run = 0; run < runsPerPolicy; run++)
            {
                status = $"{policy} przebieg {run + 1}/{runsPerPolicy}";
                yield return StartCoroutine(RunOne(policy, run, starts[run]));
            }

        Time.timeScale = 1f;
        SaveCsv();
        running = false;
        status = "gotowe";
    }

    private IEnumerator RunOne(Policy policy, int run, Pose start)
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
            case Policy.DecisionTransformer: dtInference.StartInference(); break;
            case Policy.AutoExplorer: autoExplorer.StartExploring(); break;
        }

        float t = 0f, nextSample = 0f, nextWaypoint = 0f;
        while (t < runSeconds)
        {
            if (policy == Policy.RandomWaypoint && t >= nextWaypoint)
            {
                nextWaypoint = t + decisionInterval;
                float a = Random.Range(-Mathf.PI, Mathf.PI);          // pelny okrag,
                float yaw = carTransform.eulerAngles.y * Mathf.Deg2Rad; // tak jak 36 binow DT
                float lx = Mathf.Sin(a) * randomWaypointDistance;
                float lz = Mathf.Cos(a) * randomWaypointDistance;
                float c = Mathf.Cos(yaw), s = Mathf.Sin(yaw);
                target.position = carTransform.position
                                + new Vector3(lx * c + lz * s, 0f, -lx * s + lz * c);
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
                csv.AppendLine($"{policy},{run},{t.ToString("F1", ci)},{visited.Count}," +
                               $"{(visited.Count * gridCellSize * gridCellSize).ToString("F1", ci)}," +
                               $"{distance.ToString("F2", ci)},{col}");
            }

            yield return null;
            t += Time.deltaTime;
        }

        StopAll();
        Debug.Log($"[Benchmark] {policy} run {run}: {visited.Count} komorek, " +
                  $"{distance:F1} m, {ReadCollisions() - collisionsAtStart} kolizji");
    }

    /// <summary>Zwraca 0, gdy zliczanie kolizji jest wylaczone. Dzieki temu
    /// benchmark kompiluje sie bez modyfikacji CarAgent.cs.</summary>
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
        string path = Path.Combine(dir, $"benchmark_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv");
        File.WriteAllText(path, csv.ToString());
        Debug.Log($"[Benchmark] Zapisano {path}");
    }
}