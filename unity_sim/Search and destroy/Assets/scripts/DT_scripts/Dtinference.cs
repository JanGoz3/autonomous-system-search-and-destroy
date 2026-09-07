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
    public TofScanBuffer tofScanBuffer;
    public Transform carTransform;
    public Transform target;
    public CarAgent carAgent;   // do odczytu flagi kolizji (kara -2.0 w nagrodzie)

    [Header("Decision Timing")]
    [Tooltip("Co ile sekund DT wybiera nowy waypoint. Powinno zgadzac sie z decymacja "
           + "w build_dt_dataset.py: DECIMATE = decisionInterval / rewardTickInterval. "
           + "Przy DECIMATE=15 i logowaniu 10 Hz to 1.5 s.")]
    public float decisionInterval = 1.5f;

    [Tooltip("Krok liczenia nagrody. MUSI rownac sie logIntervalSeconds z DTDataLogger "
           + "(0.1 = 10 Hz), bo return-to-go w danych jest sumowany z ta czestotliwoscia.")]
    public float rewardTickInterval = 0.1f;

    [Header("Model Config (z wydruku dt_export_to_onnx.py)")]
    public int contextLength = 20;
    [Tooltip("Liczba wartosci w SUROWYM wektorze stanu (yaw w stopniach, bez normalizacji) - "
           + "wprost z wydruku dt_export_to_onnx.py. Normalizacje i rozwiniecie yaw na sin/cos "
           + "robi juz graf ONNX. Dla wariantu pos1_scan1p_nocmd to 85, dla pos0 - 83.")]
    public int stateDim = 85;
    public int actionDim = 2;
    [Tooltip("max_ep_len z checkpointu. Timesteps sa przycinane do maxEpLen-1, inaczej "
           + "embedding wyjdzie poza zakres przy dluzszej jezdzie.")]
    public int maxEpLen = 77;
    [Tooltip("Musi zgadzac sie z ZERO_ACTIONS_IN_CONTEXT z train_dt.py. Model trenowany "
           + "z zerami nigdy nie widzial prawdziwych akcji na wejsciu.")]
    public bool zeroActionsInContext = true;

    [Tooltip("Gdy TRUE, pozycje paddingu dostaja zera i attention_mask = 0, dokladnie jak "
           + "w get_batch() podczas treningu. Gdy FALSE (domyslnie), padding powtarza "
           + "najstarszy stan z maska 1 - niezgodne z treningiem, ale bezpieczne. "
           + "WLACZAJ DOPIERO po naprawieniu maski uwagi w decision_transformer.py: "
           + "w niepoprawionej wersji w pelni zamaskowane wiersze daja softmax(-inf) = NaN.")]
    public bool useTrainingStylePadding = false;

    [Header("Wariant stanu (MUSI zgadzac sie z build_dt_dataset.py)")]
    [Tooltip("INCLUDE_POSITION. Odznacz dla wariantu bez posX/posZ - tego, ktory ma szanse "
           + "zadzialac na Teensy.")]
    public bool includePosition = true;
    [Tooltip("INCLUDE_SCAN. Dolacza scan_dist_* i scan_age_* z TofScanBuffer.")]
    public bool includeScan = true;
    [Tooltip("INCLUDE_SCAN_PITCH. Dolacza scan_pitch_*, czyli pitch kazdego pomiaru.")]
    public bool includeScanPitch = true;
    [Tooltip("EXCLUDE_POLICY_OUTPUTS. Pomija telem_0..3 - gaz, skret i oba katy kamery. "
           + "PPO wylicza je z kierunku do waypointa, czyli z ETYKIETY.")]
    public bool excludePolicyOutputs = true;
    [Tooltip("SCAN_PITCH_SCALE_DEG z build_dt_dataset.py.")]
    public float scanPitchScaleDeg = 45f;

    [Header("Return-to-go Conditioning")]
    [Tooltip("Wpisz p90 z wydruku dt_diagnose.py (sekcja BONUS). Dla obecnego datasetu to ~19. "
           + "Wartosci rzedu 100 sa 5x poza rozkladem - model dostaje wtedy embedding zwrotu, "
           + "jakiego nigdy nie widzial.")]
    public float initialTargetReturn = 19f;

    [Tooltip("Gdy FALSE (domyslnie), RTG jest stale przez caly przebieg. Gdy TRUE, maleje "
           + "o zebrana nagrode - ale przy obecnym datasecie zjezdza do zera po ~27 decyzjach "
           + "i dalej w wartosci ujemne, ktorych w danych prawie nie ma. "
           + "Sensowniejsza alternatywa to pseudoEpisodeDecisions ponizej.")]
    public bool decayReturnToGo = false;

    [Tooltip("Gdy > 0, RTG i timesteps sa resetowane co tyle decyzji i maleja liniowo "
           + "od initialTargetReturn do zera - odtwarza zaleznosc RTG/timestep z chunkow "
           + "treningowych (autoEndAfterSteps / DECIMATE, czyli ~53). 0 = wylaczone.")]
    public int pseudoEpisodeDecisions = 0;

    [Header("Reward Function (IDENTYCZNA jak w build_dt_dataset.py)")]
    public float gridCellSize = 1.0f;
    public float coverageReward = 1.0f;
    public float stepPenalty = -0.01f;
    public float collisionPenalty = -2.0f;

    [Header("Diagnostyka")]
    [Tooltip("Wymusza akcje (0, 1.5) zamiast predykcji modelu. Target MUSI wtedy pojawic sie "
           + "dokladnie PRZED maska auta.")]
    public bool debugForceForward = false;
    [Tooltip("Rzutuje waypoint na NavMesh. UWAGA: SamplePosition zwraca najblizszy punkt "
           + "siatki, nie najblizszy OSIAGALNY.")]
    public bool projectOntoNavMesh = true;
    public float navMeshSampleRadius = 1.5f;
    public bool drawGizmos = true;

    [Header("Log decyzji (diagnostyka)")]
    [Tooltip("Zapisuje kazda decyzje do CSV. Analiza: analyze_dt_run.py")]
    public bool logDecisions = true;
    public string decisionLogFolder = "DTDecisionLog";
    [Tooltip("Etykieta trafiajaca do nazwy pliku - np. nazwa polityki albo numer przebiegu.")]
    public string decisionLogTag = "";
    [Tooltip("Co ile decyzji dopisywac log na dysk. Dzieki temu crash albo wyjscie z Play Mode "
           + "nie kasuje calego przebiegu. 0 = zapis tylko na koncu.")]
    public int flushEveryDecisions = 25;

    [Header("Runtime State (read-only)")]
    [Tooltip("Ostatnia akcja modelu w ukladzie auta, w metrach: x = w prawo, z = do przodu.")]
    public Vector2 lastLocalAction;
    [Tooltip("Kat ostatniej akcji w stopniach. 0 = prosto, dodatni = w prawo.")]
    public float lastDecisionAngleDeg;
    public int waypointsOffNavMesh = 0;
    public bool isActive = false;
    public float currentReturnToGo;
    public int decisionCount = 0;
    public int cellsVisited = 0;
    [Tooltip("Ile decyzji mialo |kat| > 90 st, czyli cel ZA autem.")]
    public int decisionsBehind = 0;
    [Tooltip("Ile decyzji zapadlo, gdy auto nie ruszylo sie o wiecej niz 0.1 m "
           + "od poprzedniej decyzji.")]
    public int decisionsWhileStalled = 0;

    private Worker m_Worker;
    private float decisionTimer = 0f;
    private float rewardTimer = 0f;
    private float pendingReward = 0f;
    private int stepInPseudoEpisode = 0;

    private readonly List<float[]> stateHistory = new List<float[]>();
    private readonly List<float[]> actionHistory = new List<float[]>();
    private readonly List<float> returnToGoHistory = new List<float>();
    private readonly HashSet<Vector2Int> visitedCells = new HashSet<Vector2Int>();

    private readonly StringBuilder decisionCsv = new StringBuilder();
    private string decisionLogPath = null;
    private Vector3 lastDecisionPos;
    private float episodeTime = 0f;
    private bool stateDimWarned = false;

    void Start()
    {
        if (dtModelAsset == null)
        {
            Debug.LogError("[DTInference] Brak dtModelAsset - podepnij model ONNX.", this);
            return;
        }
        var model = ModelLoader.Load(dtModelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    public void StartInference(bool clearScanBuffer = true)
    {
        if (m_Worker == null)
        {
            Debug.LogError("[DTInference] Worker nie zainicjalizowany - nie startuje.", this);
            return;
        }

        stateDimWarned = false;
        if (includeScan && tofScanBuffer == null)
            Debug.LogError("[DTInference] includeScan = true, ale brak referencji "
                + "TofScanBuffer. Kolumny skanu beda zerami, a model dostanie dane "
                + "niezgodne z treningiem.");
        if (clearScanBuffer && tofScanBuffer != null) tofScanBuffer.Clear();

        stateHistory.Clear();
        actionHistory.Clear();
        returnToGoHistory.Clear();
        visitedCells.Clear();

        currentReturnToGo = initialTargetReturn;
        decisionTimer = 0f;
        rewardTimer = 0f;
        pendingReward = 0f;
        decisionCount = 0;
        stepInPseudoEpisode = 0;
        cellsVisited = 0;
        waypointsOffNavMesh = 0;
        decisionsBehind = 0;
        decisionsWhileStalled = 0;
        episodeTime = 0f;
        lastDecisionPos = carTransform.position;
        isActive = true;

        OpenDecisionLog();
        MakeDecision();

        Debug.Log($"[DTInference] Start. interwal={decisionInterval:F2}s  "
                + $"RTG={initialTargetReturn:F1}  "
                + $"pseudoEpizod={(pseudoEpisodeDecisions > 0 ? pseudoEpisodeDecisions.ToString() : "wyl")}");
    }

    public void StopInference()
    {
        if (!isActive) return;
        isActive = false;
        FlushDecisionLog();

        Debug.Log($"[DTInference] Koniec. decyzji={decisionCount}  " +
                  $"pozaNavMesh={waypointsOffNavMesh}  " +
                  $"celZaAutem={decisionsBehind}  " +
                  $"decyzjiWBezruchu={decisionsWhileStalled}  " +
                  $"komorek={cellsVisited}");
    }

    void OnDisable() { if (isActive) StopInference(); }
    void OnApplicationQuit() { if (isActive) StopInference(); }

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

        int guard = 0;
        while (decisionTimer >= decisionInterval && guard++ < 8)
        {
            decisionTimer -= decisionInterval;
            MakeDecision();
        }
    }

    void FixedUpdate()
    {
        if (!isActive) return;

        rewardTimer += Time.fixedDeltaTime;
        if (rewardTimer < rewardTickInterval) return;
        rewardTimer -= rewardTickInterval;

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
        float[] scanDist = tofScanBuffer != null ? tofScanBuffer.GetNormalizedDistances() : null;
        float[] scanAge = tofScanBuffer != null ? tofScanBuffer.GetNormalizedAges() : null;
        float[] scanPitch = tofScanBuffer != null ? tofScanBuffer.GetMeasurementPitchesDegrees() : null;

        float[] state = new float[stateDim];
        int k = 0;

        if (includePosition)
        {
            state[k++] = carTransform.position.x;
            state[k++] = carTransform.position.z;
        }
        state[k++] = carTransform.eulerAngles.y;

        for (int i = 0; i < telemetry.Length; i++)
        {
            if (excludePolicyOutputs && i < 4) continue;   // telem_0..3 = wyjscia PPO
            if (k >= stateDim) break;
            state[k++] = telemetry[i];
        }

        if (includeScan && scanDist != null && scanAge != null)
        {
            for (int i = 0; i < scanDist.Length && k < stateDim; i++)
                state[k++] = scanDist[i];
            for (int i = 0; i < scanAge.Length && k < stateDim; i++)
                state[k++] = scanAge[i];
            if (includeScanPitch && scanPitch != null)
                for (int i = 0; i < scanPitch.Length && k < stateDim; i++)
                    state[k++] = scanPitch[i] / scanPitchScaleDeg;
        }

        if (k != stateDim && !stateDimWarned)
        {
            stateDimWarned = true;
            Debug.LogError($"[DTInference] Zlozono {k} wartosci, a stateDim={stateDim}. "
                + "Model dostaje ZLE dane. Sprawdz stateDim, flagi wariantu i liczbe "
                + "sektorow TofScanBuffer wzgledem wydruku dt_export_to_onnx.py.");
        }
        return state;
    }

    private void UpdateReturnToGo()
    {
        if (pseudoEpisodeDecisions > 0)
        {

            if (stepInPseudoEpisode >= pseudoEpisodeDecisions) stepInPseudoEpisode = 0;
            float frac = 1f - (float)stepInPseudoEpisode / pseudoEpisodeDecisions;
            currentReturnToGo = initialTargetReturn * frac;
        }
        else if (decayReturnToGo)
        {
            currentReturnToGo -= pendingReward;
        }

        pendingReward = 0f;
    }

    private void MakeDecision()
    {
        UpdateReturnToGo();

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

        int tsNow = pseudoEpisodeDecisions > 0 ? stepInPseudoEpisode : decisionCount;

        for (int i = 0; i < contextLength; i++)
        {
            bool isPad = i < pad;

            if (isPad && useTrainingStylePadding)
            {
                for (int j = 0; j < stateDim; j++) statesTensor[0, i, j] = 0f;
                for (int j = 0; j < actionDim; j++) actionsTensor[0, i, j] = 0f;
                rtgTensor[0, i, 0] = 0f;
                timestepsTensor[0, i] = 0;
                maskTensor[0, i] = 0f;
                continue;
            }

            int histIdx = Mathf.Max(0, i - pad);

            for (int j = 0; j < stateDim; j++)
                statesTensor[0, i, j] = stateHistory[histIdx][j];
            for (int j = 0; j < actionDim; j++)
                actionsTensor[0, i, j] = zeroActionsInContext ? 0f : actionHistory[histIdx][j];

            rtgTensor[0, i, 0] = returnToGoHistory[histIdx];

            int ts = tsNow - (tlen - 1) + histIdx;
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

        if (float.IsNaN(localDx) || float.IsNaN(localDz))
        {
            Debug.LogError("[DTInference] Model zwrocil NaN. Zatrzymuje inferencje - "
                + "sprawdz maske uwagi w decision_transformer.py i eksport ONNX.", this);
            StopInference();
            statesTensor.Dispose(); actionsTensor.Dispose(); rtgTensor.Dispose();
            timestepsTensor.Dispose(); maskTensor.Dispose();
            return;
        }

        if (debugForceForward) { localDx = 0f; localDz = 1.5f; }
        lastLocalAction = new Vector2(localDx, localDz);

        if (!zeroActionsInContext)
            actionHistory[actionHistory.Count - 1] = new float[] { localDx, localDz };

        float angleDeg = Mathf.Atan2(localDx, localDz) * Mathf.Rad2Deg;
        lastDecisionAngleDeg = angleDeg;
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
        stepInPseudoEpisode++;

        statesTensor.Dispose();
        actionsTensor.Dispose();
        rtgTensor.Dispose();
        timestepsTensor.Dispose();
        maskTensor.Dispose();
    }

    private void OpenDecisionLog()
    {
        decisionCsv.Clear();
        decisionLogPath = null;
        if (!logDecisions) return;

        string dir = Path.Combine(Application.persistentDataPath, decisionLogFolder);
        Directory.CreateDirectory(dir);
        string tag = string.IsNullOrEmpty(decisionLogTag) ? "run" : SanitizeTag(decisionLogTag);
        decisionLogPath = Path.Combine(dir,
            $"decisions_{tag}_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}.csv");

        File.WriteAllText(decisionLogPath,
            "decision,t,posX,posZ,yaw,localDx,localDz,angleDeg,magM," +
            "movedSinceLast,offNavMesh,navMeshShift,rtg\n");
        Debug.Log($"[DTInference] Log decyzji -> {decisionLogPath}");
    }

    private static string SanitizeTag(string tag)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            tag = tag.Replace(c, '_');
        return tag;
    }

    private void FlushDecisionLog()
    {
        if (decisionLogPath == null || decisionCsv.Length == 0) return;
        File.AppendAllText(decisionLogPath, decisionCsv.ToString());
        decisionCsv.Clear();
    }

    private void LogDecision(Vector3 pos, float angleDeg, float magM,
                             float lx, float lz, float moved,
                             bool offNavMesh, float navShift)
    {
        if (decisionLogPath == null) return;
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

        if (flushEveryDecisions > 0 && (decisionCount + 1) % flushEveryDecisions == 0)
            FlushDecisionLog();
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

    void OnDestroy()
    {
        if (isActive) StopInference();
        m_Worker?.Dispose();
    }
}