using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.InputSystem;
using Unity.InferenceEngine;

public class DTInference : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset dtModelAsset;

    [Header("References")]
    public Chassis chassis;
    public Transform carTransform;
    public Transform target;
    public CarAgent carAgent;   // do odczytu flagi kolizji (kara -2.0 w nagrodzie)

    [Header("Decision Timing")]
    [Tooltip("Co ile sekund DT wybiera nowy waypoint. MUSI zgadzac sie z decymacja w build_dt_dataset.py: DECIMATE = decisionInterval / rewardTickInterval.")]
    public float decisionInterval = 1.5f;

    [Tooltip("Krok liczenia nagrody. MUSI rownac sie logIntervalSeconds z DTDataLogger (0.1 = 10 Hz), bo return-to-go w danych jest sumowany z ta czestotliwoscia.")]
    public float rewardTickInterval = 0.1f;

    [Header("Model Config (z wydruku export_to_onnx.py)")]
    public int contextLength = 20;
    public int stateDim = 20;
    public int actionDim = 2;
    [Tooltip("max_ep_len z checkpointu. Timesteps sa przycinane do maxEpLen-1, inaczej embedding wyjdzie poza zakres przy dluzszej jezdzie.")]
    public int maxEpLen = 77;
    [Tooltip("Musi zgadzac sie z ZERO_ACTIONS_IN_CONTEXT z train_dt.py. Model trenowany z zerami nigdy nie widzial prawdziwych akcji na wejsciu.")]
    public bool zeroActionsInContext = true;

    [Header("Return-to-go Conditioning")]
    [Tooltip("UWAGA: przy etykietach eksperta sumy nagrod sa DODATNIE (33..118 w zbadanych epizodach), a RTG=0 wystepuje wylacznie na KONCU epizodu. Zostawienie 0 warunkuje model na zachowanie terminalne. Wpisz ok. p90 z wydruku build_dt_dataset.py.")]
    public float initialTargetReturn = 100f;

    [Header("Reward Function (IDENTYCZNA jak w build_dt_dataset.py)")]
    public float gridCellSize = 1.0f;
    public float coverageReward = 1.0f;
    public float stepPenalty = -0.01f;
    public float collisionPenalty = -2.0f;

    [Header("Diagnostyka")]
    [Tooltip("Wymusza akcje (0, 1.5) zamiast predykcji modelu. Target MUSI wtedy pojawic sie dokladnie PRZED maska auta.")]
    public bool debugForceForward = false;
    [Tooltip("Rzutuje waypoint na NavMesh. UWAGA: SamplePosition zwraca najblizszy punkt siatki, nie najblizszy OSIAGALNY - dla celu w scianie zwykle laduje przy jej powierzchni.")]
    public bool projectOntoNavMesh = true;
    public float navMeshSampleRadius = 1.5f;
    public bool drawGizmos = true;

    [Header("Log decyzji (diagnostyka)")]
    [Tooltip("Zapisuje kazda decyzje do CSV: kat i dlugosc waypointa, pozycje, dystans przejechany od poprzedniej decyzji, przesuniecie przez NavMesh. Analiza: analyze_dt_run.py")]
    public bool logDecisions = true;
    public string decisionLogFolder = "DTDecisionLog";
    [Tooltip("Etykieta trafiajaca do nazwy pliku - np. nazwa polityki albo numer przebiegu.")]
    public string decisionLogTag = "";

    [Header("Runtime State (read-only)")]
    [Tooltip("Ostatnia akcja modelu w ukladzie auta, w metrach: x = w prawo, z = do przodu.")]
    public Vector2 lastLocalAction;
    public int waypointsOffNavMesh = 0;
    public bool isActive = false;
    public float currentReturnToGo;
    public int decisionCount = 0;
    public int cellsVisited = 0;
    [Tooltip("Ile decyzji mialo |kat| > 90 st, czyli cel ZA autem. Driver PPO trenowal na celach 1 m przed autem w stozku +/-45 st.")]
    public int decisionsBehind = 0;
    [Tooltip("Ile decyzji zapadlo, gdy auto nie ruszylo sie o wiecej niz 0.1 m od poprzedniej decyzji.")]
    public int decisionsWhileStalled = 0;

    private Worker m_Worker;
    private float decisionTimer = 0f;
    private float rewardTimer = 0f;
    private float pendingReward = 0f;

    private readonly List<float[]> stateHistory = new List<float[]>();
    private readonly List<float[]> actionHistory = new List<float[]>();
    private readonly List<float> returnToGoHistory = new List<float>();
    private readonly HashSet<Vector2Int> visitedCells = new HashSet<Vector2Int>();

    private readonly StringBuilder decisionCsv = new StringBuilder();
    private Vector3 lastDecisionPos;
    private float episodeTime = 0f;

    void Start()
    {
        var model = ModelLoader.Load(dtModelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    public void StartInference()
    {
        stateHistory.Clear();
        actionHistory.Clear();
        returnToGoHistory.Clear();
        visitedCells.Clear();

        currentReturnToGo = initialTargetReturn;
        decisionTimer = 0f;
        rewardTimer = 0f;
        pendingReward = 0f;
        decisionCount = 0;
        cellsVisited = 0;
        waypointsOffNavMesh = 0;
        decisionsBehind = 0;
        decisionsWhileStalled = 0;
        episodeTime = 0f;
        lastDecisionPos = carTransform.position;
        isActive = true;

        decisionCsv.Clear();
        decisionCsv.AppendLine("decision,t,posX,posZ,yaw,localDx,localDz,angleDeg,magM," +
                               "movedSinceLast,offNavMesh,navMeshShift,rtg");

        MakeDecision();   // pierwszy waypoint natychmiast, bez czekania
        Debug.Log($"[DTInference] Start. initialTargetReturn={initialTargetReturn}");
    }

    public void StopInference()
    {
        isActive = false;
        if (logDecisions) SaveDecisionLog();

        Debug.Log($"[DTInference] Koniec. decyzji={decisionCount}  " +
                  $"pozaNavMesh={waypointsOffNavMesh}  " +
                  $"celZaAutem={decisionsBehind}  " +
                  $"decyzjiWBezruchu={decisionsWhileStalled}  " +
                  $"komorek={cellsVisited}");
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null)
        {
            if (kb.iKey.wasPressedThisFrame && !isActive) StartInference();
            if (kb.oKey.wasPressedThisFrame && isActive) StopInference();
        }
        if (!isActive) return;

        episodeTime += Time.deltaTime;
        decisionTimer += Time.deltaTime;
        if (decisionTimer >= decisionInterval)
        {
            decisionTimer = 0f;
            MakeDecision();
        }
    }

    void FixedUpdate()
    {
        if (!isActive) return;

        rewardTimer += Time.fixedDeltaTime;
        if (rewardTimer < rewardTickInterval) return;
        rewardTimer = 0f;

        float r = stepPenalty;

        Vector2Int cell = new Vector2Int(
            Mathf.FloorToInt(carTransform.position.x / gridCellSize),
            Mathf.FloorToInt(carTransform.position.z / gridCellSize));
        if (visitedCells.Add(cell))
        {
            r += coverageReward;
            cellsVisited = visitedCells.Count;
        }

        if (carAgent != null && carAgent.hadCollisionThisStep)
        {
            r += collisionPenalty;
            carAgent.hadCollisionThisStep = false;
        }

        pendingReward += r;
    }

    private float[] GetCurrentStateVector()
    {
        float[] telemetry = chassis.GetTelemetryState();
        float[] state = new float[stateDim];
        state[0] = carTransform.position.x;
        state[1] = carTransform.position.z;
        state[2] = carTransform.eulerAngles.y;
        for (int i = 0; i < telemetry.Length && 3 + i < stateDim; i++)
            state[3 + i] = telemetry[i];
        return state;
    }

    private void MakeDecision()
    {
        // currentReturnToGo -= pendingReward;   // dekrementacja wylaczona swiadomie
        pendingReward = 0f;

        stateHistory.Add(GetCurrentStateVector());
        returnToGoHistory.Add(currentReturnToGo);
        actionHistory.Add(new float[actionDim]);

        while (stateHistory.Count > contextLength)
        {
            stateHistory.RemoveAt(0);
            actionHistory.RemoveAt(0);
            returnToGoHistory.RemoveAt(0);
        }

        int tlen = stateHistory.Count;
        int pad = contextLength - tlen;

        var statesTensor = new Tensor<float>(new TensorShape(1, contextLength, stateDim));
        var actionsTensor = new Tensor<float>(new TensorShape(1, contextLength, actionDim));
        var rtgTensor = new Tensor<float>(new TensorShape(1, contextLength, 1));
        var timestepsTensor = new Tensor<int>(new TensorShape(1, contextLength));
        var maskTensor = new Tensor<float>(new TensorShape(1, contextLength));

        for (int i = 0; i < contextLength; i++)
        {
            int histIdx = Mathf.Max(0, i - pad);

            for (int j = 0; j < stateDim; j++)
                statesTensor[0, i, j] = stateHistory[histIdx][j];
            for (int j = 0; j < actionDim; j++)
                actionsTensor[0, i, j] = zeroActionsInContext ? 0f : actionHistory[histIdx][j];

            rtgTensor[0, i, 0] = returnToGoHistory[histIdx];

            int ts = decisionCount - (tlen - 1) + histIdx;
            timestepsTensor[0, i] = Mathf.Clamp(ts, 0, maxEpLen - 1);
            maskTensor[0, i] = 1f;
        }

        m_Worker.SetInput("states", statesTensor);
        m_Worker.SetInput("actions", actionsTensor);
        m_Worker.SetInput("returns_to_go", rtgTensor);
        m_Worker.SetInput("timesteps", timestepsTensor);
        m_Worker.SetInput("attention_mask", maskTensor);
        m_Worker.Schedule();

        var outputTensor = m_Worker.PeekOutput("predicted_action") as Tensor<float>;
        float[] predicted = outputTensor.DownloadToArray();   // [localDx, localDz] w METRACH

        float localDx = predicted[0];
        float localDz = predicted[1];

        if (debugForceForward) { localDx = 0f; localDz = 1.5f; }
        lastLocalAction = new Vector2(localDx, localDz);

        if (!zeroActionsInContext)
            actionHistory[actionHistory.Count - 1] = new float[] { localDx, localDz };

        // --- diagnostyka: kat, dlugosc, ruch od poprzedniej decyzji -----------
        float angleDeg = Mathf.Atan2(localDx, localDz) * Mathf.Rad2Deg;
        float magM = new Vector2(localDx, localDz).magnitude;
        Vector3 nowPos = carTransform.position;
        float moved = Vector3.Distance(new Vector3(nowPos.x, 0f, nowPos.z),
                                       new Vector3(lastDecisionPos.x, 0f, lastDecisionPos.z));
        lastDecisionPos = nowPos;

        if (Mathf.Abs(angleDeg) > 90f) decisionsBehind++;
        if (decisionCount > 0 && moved < 0.1f) decisionsWhileStalled++;

        Vector3 worldOffset = Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f)
                              * new Vector3(localDx, 0f, localDz);
        Vector3 desired = carTransform.position + worldOffset;

        bool offNavMesh = false;
        float navShift = 0f;

        if (projectOntoNavMesh)
        {
            if (NavMesh.SamplePosition(desired, out NavMeshHit navHit,
                                       navMeshSampleRadius, NavMesh.AllAreas))
            {
                navShift = Vector3.Distance(desired, navHit.position);
                desired = navHit.position;
            }
            else
            {
                offNavMesh = true;
                waypointsOffNavMesh++;   // cel poza zasiegiem NavMesh - zostawiamy poprzedni
            }
        }

        if (!offNavMesh)
            target.position = desired + new Vector3(0, 0.05f, 0);

        LogDecision(nowPos, angleDeg, magM, localDx, localDz, moved, offNavMesh, navShift);

        decisionCount++;

        statesTensor.Dispose();
        actionsTensor.Dispose();
        rtgTensor.Dispose();
        timestepsTensor.Dispose();
        maskTensor.Dispose();
    }

    private void LogDecision(Vector3 pos, float angleDeg, float magM,
                             float lx, float lz, float moved,
                             bool offNavMesh, float navShift)
    {
        if (!logDecisions) return;
        var ci = CultureInfo.InvariantCulture;
        decisionCsv.Append(decisionCount.ToString(ci)).Append(',')
            .Append(episodeTime.ToString("F2", ci)).Append(',')
            .Append(pos.x.ToString("F3", ci)).Append(',')
            .Append(pos.z.ToString("F3", ci)).Append(',')
            .Append(carTransform.eulerAngles.y.ToString("F2", ci)).Append(',')
            .Append(lx.ToString("F4", ci)).Append(',')
            .Append(lz.ToString("F4", ci)).Append(',')
            .Append(angleDeg.ToString("F2", ci)).Append(',')
            .Append(magM.ToString("F3", ci)).Append(',')
            .Append(moved.ToString("F3", ci)).Append(',')
            .Append(offNavMesh ? "1" : "0").Append(',')
            .Append(navShift.ToString("F3", ci)).Append(',')
            .Append(currentReturnToGo.ToString("F2", ci))
            .AppendLine();
    }

    private void SaveDecisionLog()
    {
        if (decisionCsv.Length == 0) return;
        string dir = Path.Combine(Application.persistentDataPath, decisionLogFolder);
        Directory.CreateDirectory(dir);
        string tag = string.IsNullOrEmpty(decisionLogTag) ? "run" : decisionLogTag;
        string path = Path.Combine(dir,
            $"decisions_{tag}_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}.csv");
        File.WriteAllText(path, decisionCsv.ToString());
        Debug.Log($"[DTInference] Zapisano log decyzji: {path}");
    }

    void OnDrawGizmos()
    {
        if (!drawGizmos || !isActive || carTransform == null || target == null) return;

        Gizmos.color = Color.green;                       // waypoint modelu
        Gizmos.DrawLine(carTransform.position, target.position);
        Gizmos.DrawWireSphere(target.position, 0.25f);

        Gizmos.color = Color.cyan;                        // kierunek jazdy auta
        Gizmos.DrawRay(carTransform.position,
                       Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f) * Vector3.forward * 1.5f);
    }

    void OnDestroy() { m_Worker?.Dispose(); }
}