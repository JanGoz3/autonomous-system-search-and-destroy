using System.Collections.Generic;
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
    [Tooltip("Dobierz z wydruku build_dt_dataset.py (kolumna 'suma nagrod'). Returny sa teraz w wiekszosci UJEMNE - nie zostawiaj 50, to ekstrapolacja daleko poza obserwowany zakres.")]
    public float initialTargetReturn = 0f;

    [Header("Reward Function (IDENTYCZNA jak w build_dt_dataset.py)")]
    public float gridCellSize = 1.0f;
    public float coverageReward = 1.0f;
    public float stepPenalty = -0.01f;
    public float collisionPenalty = -2.0f;

    [Header("Diagnostyka")]
    [Tooltip("Wymusza akcje (0, 1.5) zamiast predykcji modelu. Target MUSI wtedy pojawic sie dokladnie PRZED maska auta. Jesli pojawia sie z boku lub z tylu - blad jest w konwersji ukladu, nie w modelu.")]
    public bool debugForceForward = false;
    [Tooltip("Rzutuje waypoint na NavMesh. Model nie ma pojecia o scianach - 42% jego predykcji jest odchylonych o wiecej niz 45 st, wiec czesc celow ladowalaby w geometrii.")]
    public bool projectOntoNavMesh = true;
    public float navMeshSampleRadius = 1.5f;
    public bool drawGizmos = true;

    [Header("Runtime State (read-only)")]
    [Tooltip("Ostatnia akcja modelu w ukladzie auta, w metrach: x = w prawo, z = do przodu.")]
    public Vector2 lastLocalAction;
    public int waypointsOffNavMesh = 0;
    public bool isActive = false;
    public float currentReturnToGo;
    public int decisionCount = 0;
    public int cellsVisited = 0;

    private Worker m_Worker;
    private float decisionTimer = 0f;
    private float rewardTimer = 0f;
    private float pendingReward = 0f;

    private readonly List<float[]> stateHistory = new List<float[]>();
    private readonly List<float[]> actionHistory = new List<float[]>();
    private readonly List<float> returnToGoHistory = new List<float>();
    private readonly HashSet<Vector2Int> visitedCells = new HashSet<Vector2Int>();

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
        isActive = true;

        MakeDecision();   // pierwszy waypoint natychmiast, bez czekania
        Debug.Log($"[DTInference] Start. initialTargetReturn={initialTargetReturn}");
    }

    public void StopInference() { isActive = false; }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null)
        {
            if (kb.iKey.wasPressedThisFrame && !isActive) StartInference();
            if (kb.oKey.wasPressedThisFrame && isActive) StopInference();
        }
        if (!isActive) return;

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
//        currentReturnToGo -= pendingReward;
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

        Vector3 worldOffset = Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f)
                              * new Vector3(localDx, 0f, localDz);
        Vector3 desired = carTransform.position + worldOffset;

        if (projectOntoNavMesh)
        {
            if (NavMesh.SamplePosition(desired, out NavMeshHit navHit,
                                       navMeshSampleRadius, NavMesh.AllAreas))
            {
                desired = navHit.position;
            }
            else
            {
                waypointsOffNavMesh++;   // cel poza zasiegiem NavMesh - zostawiamy poprzedni
                decisionCount++;
                statesTensor.Dispose(); actionsTensor.Dispose(); rtgTensor.Dispose();
                timestepsTensor.Dispose(); maskTensor.Dispose();
                return;
            }
        }

        target.position = desired + new Vector3(0, 0.05f, 0);
        decisionCount++;

        statesTensor.Dispose();
        actionsTensor.Dispose();
        rtgTensor.Dispose();
        timestepsTensor.Dispose();
        maskTensor.Dispose();
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