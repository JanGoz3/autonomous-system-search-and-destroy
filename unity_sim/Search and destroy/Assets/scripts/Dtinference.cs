using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using Unity.InferenceEngine;


public class DTInference : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset dtModelAsset;

    [Header("References")]
    public Chassis chassis;
    public Transform carTransform;
    public Transform target; // ten sam obiekt, ktorego uzywa CarAgent / AutoExplorer

    [Header("Decision Timing")]
    [Tooltip("Co ile sekund DT podejmuje nowa decyzje o waypoincie. Powinno byc RZADZIEJ niz driver steruje (driver dziala co FixedUpdate).")]
    public float decisionInterval = 1.5f;

    [Header("Model Config (MUSI zgadzac sie z konfiguracja treningowa)")]
    public int contextLength = 20;
    public int stateDim = 20;
    public int actionDim = 2;

    [Header("Return-to-go Conditioning")]
    [Tooltip("Poczatkowy target return - im wyzszy, tym model probuje 'nasladowac' lepsze zademonstrowane epizody. Dobierz na podstawie sum nagrod z build_dataset.py.")]
    public float initialTargetReturn = 50f;

    [Header("Reward Function (MUSI byc IDENTYCZNE jak w build_dataset.py)")]
    public float gridCellSize = 1.0f;
    public float coverageReward = 1.0f;
    public float stepPenalty = -0.01f;

    [Header("Runtime State (read-only)")]
    public bool isActive = false;
    public float currentReturnToGo;
    public int decisionCount = 0;

    private Worker m_Worker;
    private float timer = 0f;

    private List<float[]> stateHistory = new List<float[]>();
    private List<float[]> actionHistory = new List<float[]>();
    private List<float> returnToGoHistory = new List<float>();

    private HashSet<Vector2Int> visitedCells = new HashSet<Vector2Int>();

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
        timer = 0f;
        decisionCount = 0;
        isActive = true;

        Debug.Log($"[DTInference] Started. initialTargetReturn={initialTargetReturn}");
    }

    public void StopInference()
    {
        isActive = false;
    }

    void Update()
    {
        var keyboard = Keyboard.current;
        if (keyboard != null)
        {
            if (keyboard.iKey.wasPressedThisFrame && !isActive) StartInference();
            if (keyboard.oKey.wasPressedThisFrame && isActive) StopInference();
        }

        if (!isActive) return;

        timer += Time.deltaTime;
        if (timer < decisionInterval) return;
        timer = 0f;

        MakeDecision();
    }

    private float ComputeStepReward()
    {

        Vector2Int cell = new Vector2Int(
            Mathf.FloorToInt(carTransform.position.x / gridCellSize),
            Mathf.FloorToInt(carTransform.position.z / gridCellSize)
        );

        float r = stepPenalty;
        if (!visitedCells.Contains(cell))
        {
            visitedCells.Add(cell);
            r += coverageReward;
        }
        return r;
    }

    private float[] GetCurrentStateVector()
    {

        float[] telemetry = chassis.GetTelemetryState();

        float[] state = new float[stateDim];
        state[0] = carTransform.position.x;
        state[1] = carTransform.position.z;
        state[2] = carTransform.eulerAngles.y;
        for (int i = 0; i < telemetry.Length; i++)
        {
            state[3 + i] = telemetry[i];
        }
        return state;
    }

    private void MakeDecision()
    {

        if (decisionCount > 0)
        {
            float reward = ComputeStepReward();
            currentReturnToGo -= reward;
        }

        float[] currentState = GetCurrentStateVector();

        stateHistory.Add(currentState);
        returnToGoHistory.Add(currentReturnToGo);
        actionHistory.Add(new float[actionDim]);

        if (stateHistory.Count > contextLength)
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
            if (i < pad)
            {
                for (int j = 0; j < stateDim; j++) statesTensor[0, i, j] = 0f;
                for (int j = 0; j < actionDim; j++) actionsTensor[0, i, j] = 0f;
                rtgTensor[0, i, 0] = 0f;
                timestepsTensor[0, i] = 0;
                maskTensor[0, i] = 0f;
            }
            else
            {
                int histIdx = i - pad;
                for (int j = 0; j < stateDim; j++) statesTensor[0, i, j] = stateHistory[histIdx][j];
                for (int j = 0; j < actionDim; j++) actionsTensor[0, i, j] = actionHistory[histIdx][j];
                rtgTensor[0, i, 0] = returnToGoHistory[histIdx];
                timestepsTensor[0, i] = decisionCount - (tlen - 1) + histIdx;
                maskTensor[0, i] = 1f;
            }
        }

        m_Worker.SetInput("states", statesTensor);
        m_Worker.SetInput("actions", actionsTensor);
        m_Worker.SetInput("returns_to_go", rtgTensor);
        m_Worker.SetInput("timesteps", timestepsTensor);
        m_Worker.SetInput("attention_mask", maskTensor);
        m_Worker.Schedule();

        var outputTensor = m_Worker.PeekOutput("predicted_action") as Tensor<float>;
        float[] predicted = outputTensor.DownloadToArray(); // [localDx, localDz], juz w METRACH

        float localDx = predicted[0];
        float localDz = predicted[1];

        actionHistory[actionHistory.Count - 1] = new float[] { localDx / 20f, localDz / 20f };

        float yawRad = carTransform.eulerAngles.y * Mathf.Deg2Rad;
        float worldDx = localDx * Mathf.Cos(yawRad) - localDz * Mathf.Sin(yawRad);
        float worldDz = localDx * Mathf.Sin(yawRad) + localDz * Mathf.Cos(yawRad);

        Vector3 newTargetPos = carTransform.position + new Vector3(worldDx, 0f, worldDz);
        target.position = newTargetPos;

        decisionCount++;

        statesTensor.Dispose();
        actionsTensor.Dispose();
        rtgTensor.Dispose();
        timestepsTensor.Dispose();
        maskTensor.Dispose();
    }

    void OnDestroy()
    {
        m_Worker?.Dispose();
    }
}