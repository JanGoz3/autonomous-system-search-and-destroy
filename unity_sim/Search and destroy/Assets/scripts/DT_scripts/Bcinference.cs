using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.InputSystem;
using Unity.InferenceEngine;

/// <summary>
/// Inferencja WaypointTransformera (droga A - behavior cloning).
///
/// Zastepuje DTInference. Rozne wzgledem niego:
///   * BRAK return-to-go i timestepow - model ich nie ma
///   * padding IDENTYCZNY jak w treningu: zera + attention_mask = 0. Stara
///     flaga useTrainingStylePadding istniala tylko dlatego, ze maska uwagi
///     generowala NaN; po jej naprawie nie ma juz wyboru do zrobienia.
///   * BRAK liczenia nagrody - nie jest do niczego potrzebna
///   * BRAMKA STOZKA KIERUNKU - model zwraca pelny rozklad po kubelkach, wiec
///     wybieramy najlepszy kubelek MIESZCZACY SIE w wykonalnym stozku. Auto z
///     kierownica Ackermanna nie potrafi wykonac komendy "za siebie".
///   * BRAMKA PEWNOSCI - przy niskiej pewnosci trzymamy poprzedni waypoint
///     zamiast skakac w losowa strone.
///
/// Klawisze: I = start, O = stop.
/// </summary>
public class BCInference : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset modelAsset;

    [Header("References")]
    public Chassis chassis;
    public TofScanBuffer tofScanBuffer;
    public Transform carTransform;
    public Transform target;
    [Tooltip("Potrzebny do przejecia sterowania przy wychodzeniu z zaklinowania.")]
    public CarAgent carAgent;

    [Header("Model Config (z wydruku export_bc_to_onnx.py)")]
    public int contextLength = 20;
    [Tooltip("SUROWY wektor stanu, yaw w stopniach. Normalizacje i sin/cos robi graf ONNX.")]
    public int stateDim = 56;
    public int nDirBins = 36;

    [Header("Wariant stanu (MUSI zgadzac sie z build_bc_dataset.py)")]
    [Tooltip("Jawny kanal kierunku jazdy jako PIERWSZA wartosc stanu. MUSI zgadzac sie "
           + "z ADD_DIRECTION_CHANNEL w build_bc_dataset.py.")]
    public bool addDirectionChannel = true;
    [Tooltip("+1 = trasa w kierunku podstawowym (pliki bc_fwd_*), -1 = odwrocona "
           + "(bc_rev_*). Bez tego kanalu model mial w tym samym miejscu dwa poprawne "
           + "rozwiazania i potrafil przeskoczyc na drugie w trakcie zakretu - w logu "
           + "widac bylo zawrotki po ~150 st z pewnoscia powyzej 0.95.")]
    public float drivingDirection = 1f;
    public bool includePosition = false;
    public bool includeScan = true;
    public bool includeScanPitch = true;
    [Tooltip("Pomija telem_0..3 - wyjscia polityki PPO.")]
    public bool excludePolicyOutputs = true;
    [Tooltip("Ile slotow detekcji YOLO wlaczyc do stanu. 0 = bez YOLO. MUSI rownac sie "
           + "YOLO_MAX_SLOTS w build_bc_dataset.py. Slot 2 odpala sie w 0.3% krokow i "
           + "generuje ekstrema po normalizacji, wiec 2 to rozsadne maksimum.")]
    public int yoloMaxSlots = 2;
    public int yoloFirstIndex = 11;
    public int yoloFeaturesPerSlot = 9;
    [Tooltip("Ktore cechy w slocie pominac. Uklad slotu: 0=x 1=y 2=w 3=h 4=conf "
           + "5=chair 6=door 7=person 8=target. Flagi PERSON i TARGET sa w zadaniu "
           + "okrazenia zawsze zerowe. MUSI rownac sie YOLO_DROP_OFFSETS w Pythonie.")]
    public int[] yoloDropOffsets = new int[] { 7, 8 };
    public float scanPitchScaleDeg = 45f;

    [Header("Decision Timing")]
    [Tooltip("Co ile sekund model wybiera nowy waypoint. Musi rownac sie "
           + "DECIMATE * logIntervalSeconds z etapu zbierania danych (10 * 0.1 = 1.0 s).")]
    public float decisionInterval = 1.0f;

    [Header("Bramka stozka kierunku")]
    [Tooltip("Maksymalny |kat| komendy w stopniach. Powinien rownac sie "
           + "MAX_LABEL_ANGLE_DEG z build_bc_dataset.py. Kubelki poza stozkiem sa "
           + "pomijane i wybierany jest najlepszy kubelek w srodku.")]
    public float maxCommandAngleDeg = 70f;

    [Tooltip("Zamiast twardego argmaxa liczy srednia kolowa po kubelkach w stozku, "
           + "wazona prawdopodobienstwem. Usuwa kwantyzacje 360/nDirBins stopni.")]
    public bool useSoftDirection = true;
    [Tooltip("Ile najlepszych kubelkow uwzglednic w sredniej kolowej.")]
    public int softTopK = 5;

    [Header("Martwa strefa na prostych")]
    [Tooltip("Kat mniejszy niz tyle jest zerowany. W walidacji model mial w kubelku "
           + "0-10 st blad 15.3 st przy 5.7 st polityki 'zawsze prosto' - dorzucal szum "
           + "tam, gdzie wystarczy jechac na wprost. 0 = wylaczone.")]
    public float straightDeadzoneDeg = 8f;

    [Header("Bramka pewnosci")]
    [Tooltip("Jesli masa prawdopodobienstwa w stozku spadnie ponizej tej wartosci, "
           + "waypoint NIE jest zmieniany. Przy krotkiej historii na starcie model "
           + "ma pewnosc rzedu 0.2-0.3, wiec pierwsze decyzje sa naturalnie wstrzymane.")]
    public float minConeMass = 0.35f;

    [Header("Dlugosc waypointa")]
    public float minWaypointDistance = 0.6f;
    public float maxWaypointDistance = 2.5f;

    [Header("Wychodzenie z zaklinowania")]
    [Tooltip("Stozek +/-70 st nie potrafi wyrazic komendy 'cofaj', a Motor.SetSpeed przy "
           + "ujemnej wartosci i jazdzie w przod HAMUJE. Dodatkowo PPO uczylo sie na celach "
           + "spawnowanych w +/-45 st PRZED autem, wiec celu za soba nigdy nie widzialo. "
           + "Manewr musi wiec omijac i model, i PPO - steruje chassis bezposrednio.")]
    public bool recoveryEnabled = true;
    [Tooltip("Po ilu kolejnych decyzjach w bezruchu wlaczyc manewr.")]
    public int stallThresholdDecisions = 2;
    [Tooltip("Ruch ponizej tej wartosci miedzy decyzjami liczy sie jako bezruch.")]
    public float stallDistanceM = 0.1f;
    public float reverseSeconds = 1.6f;
    [Range(-1f, 0f)] public float reverseSpeed = -0.7f;
    [Tooltip("Skret przy cofaniu, przeciwny do ostatniej komendy - auto cofa sie "
           + "'odkrecajac' od przeszkody. 0 = cofanie na prosto.")]
    [Range(0f, 1f)] public float reverseSteer = 0.7f;
    [Tooltip("Ile sekund po manewrze nie liczyc bezruchu, zeby nie wpadac w petle.")]
    public float recoveryCooldownSeconds = 3f;

    public enum TraversabilityMode
    {
        Wylaczony,   // brak filtru - zachowanie identyczne w symulacji i na sprzecie
        NavMesh,     // NavMesh.Raycast - TYLKO SYMULACJA, na robocie nie ma siatki
        ToF          // profil z TofScanBuffer - przenoszalne na sprzet
    }

    [Header("Filtr przejezdnosci")]
    [Tooltip("NavMesh: dokladny, ale korzysta z wiedzy o mapie, ktorej robot nie ma. "
           + "Dodatkowo siatka konczy sie o promien agenta przed geometria (R=0.23), "
           + "wiec waypoint w korytarzu bywa odrzucany fałszywie.\n"
           + "ToF: ten sam test na profilu z czujnika pokladowego - dziala tak samo w "
           + "symulacji i na robocie, ale widzi tylko tam, gdzie wiezyczka zdazyla "
           + "zmierzyc (okolo 11 z 16 sektorow).\n"
           + "Wylaczony: bez zabezpieczenia, ochrone daje tylko manewr wychodzenia "
           + "z zaklinowania (ten jest w pelni przenoszalny).")]
    public TraversabilityMode traversabilityMode = TraversabilityMode.NavMesh;
    [Tooltip("Odstep od wykrytej przeszkody przy skracaniu waypointa.")]
    public float traversabilityMargin = 0.25f;

    [Header("Filtr ToF (gdy tryb = ToF)")]
    [Tooltip("Polowa szerokosci wachlarza sektorow branych pod uwage wokol kierunku "
           + "waypointa. Sektor ma 360/(zakres yaw) stopni; wieksza wartosc = bardziej "
           + "zachowawczo, bo bierzemy MINIMUM odleglosci z wachlarza.")]
    public float tofConeHalfWidthDeg = 12f;
    [Tooltip("Zasieg czujnika w metrach. MUSI zgadzac sie z TofScanBuffer.MaxDistanceMm.")]
    public float tofMaxRangeM = 3.0f;
    [Tooltip("Pomijaj sektory, w ktorych wiezyczka byla odchylona w pionie bardziej niz "
           + "tyle stopni - taki pomiar nie mowi o przeszkodzie na poziomie jazdy. "
           + "90 = bierz wszystkie.")]
    public float tofMaxAbsPitchDeg = 90f;

    [Header("NavMesh")]
    [Tooltip("Rzut waypointa na siatke. To rowniez wiedza o mapie - przy transferze "
           + "na sprzet odznacz.")]
    public bool projectOntoNavMesh = true;
    [Tooltip("0.3, nie 1.5: przy duzym promieniu rzut przeskakuje przez sciany.")]
    public float navMeshSampleRadius = 0.3f;

    [Header("Diagnostyka")]
    public bool debugForceForward = false;
    public bool drawGizmos = true;
    public bool logDecisions = true;
    public string decisionLogFolder = "BCDecisionLog";
    public string decisionLogTag = "";
    public int flushEveryDecisions = 25;

    [Header("Runtime (read-only)")]
    public bool isActive = false;
    public int decisionCount = 0;
    public Vector2 lastLocalAction;
    public float lastAngleDeg;
    public float lastMagnitudeM;
    public float lastConeMass;
    [Tooltip("Ile razy surowy argmax modelu wskazal POZA stozek. Wysoka wartosc "
           + "oznacza, ze w danych treningowych zostaly jeszcze zle etykiety.")]
    public int argmaxOutsideCone = 0;
    [Tooltip("Ile decyzji wstrzymano z powodu niskiej pewnosci.")]
    public int decisionsHeld = 0;
    public int waypointsOffNavMesh = 0;
    [Tooltip("Ile razy waypoint byl SKRACANY, bo miedzy autem a nim byla przegroda. "
           + "Wysoka wartosc = model wskazuje w sciany, czyli slabo radzi sobie z narozníkami.")]
    public int waypointsShortened = 0;
    [Tooltip("Tryb ToF: ile razy w kierunku waypointa NIE BYLO swiezego pomiaru, wiec "
           + "filtr przepuscil bez sprawdzenia. Na robocie bylo by tak samo.")]
    public int tofNoMeasurement = 0;
    [Tooltip("Ile decyzji zapadlo, gdy auto nie ruszylo sie o wiecej niz stallDistanceM.")]
    public int decisionsWhileStalled = 0;
    [Tooltip("Ile razy odpalil manewr wychodzenia z zaklinowania.")]
    public int recoveryCount = 0;
    public bool isRecovering = false;

    private Worker m_Worker;
    private float m_DecisionTimer;
    private float m_EpisodeTime;
    private bool m_StateDimWarned;

    private readonly List<float[]> m_StateHistory = new List<float[]>();
    private readonly StringBuilder m_Csv = new StringBuilder();
    private string m_LogPath;
    private Vector3 m_LastDecisionPos;
    private int m_StalledStreak;
    private float m_RecoveryTimer;
    private float m_CooldownTimer;
    private float m_RecoverySteer;

    // ------------------------------------------------------------------

    void Reset()
    {
        chassis = GetComponent<Chassis>();
        tofScanBuffer = GetComponentInChildren<TofScanBuffer>(true);
        carTransform = transform;
        carAgent = GetComponent<CarAgent>();
    }

    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("[BCInference] Brak modelAsset - podepnij WaypointTransformer.onnx.", this);
            return;
        }
        m_Worker = new Worker(ModelLoader.Load(modelAsset), BackendType.GPUCompute);
    }

    public void StartInference(bool clearScanBuffer = true)
    {
        if (m_Worker == null)
        {
            Debug.LogError("[BCInference] Worker nie zainicjalizowany.", this);
            return;
        }
        if (includeScan && tofScanBuffer == null)
            Debug.LogError("[BCInference] includeScan = true, ale brak TofScanBuffer.", this);
        if (target == null || carTransform == null || chassis == null)
        {
            Debug.LogError("[BCInference] Brak referencji target/carTransform/chassis.", this);
            return;
        }

        if (clearScanBuffer && tofScanBuffer != null) tofScanBuffer.Clear();

        m_StateHistory.Clear();
        m_StateDimWarned = false;
        m_DecisionTimer = 0f;
        m_EpisodeTime = 0f;
        decisionCount = 0;
        argmaxOutsideCone = 0;
        decisionsHeld = 0;
        waypointsOffNavMesh = 0;
        waypointsShortened = 0;
        tofNoMeasurement = 0;
        decisionsWhileStalled = 0;
        m_LastDecisionPos = carTransform.position;
        m_StalledStreak = 0;
        m_RecoveryTimer = 0f;
        m_CooldownTimer = 0f;
        recoveryCount = 0;
        isRecovering = false;
        isActive = true;

        OpenLog();
        MakeDecision();

        Debug.Log($"[BCInference] Start. interwal={decisionInterval:F2}s  K={contextLength}  "
                + $"stateDim={stateDim}  stozek=+/-{maxCommandAngleDeg:F0} st");
    }

    public void StopInference()
    {
        if (!isActive) return;
        isActive = false;
        EndRecovery();
        FlushLog();
        Debug.Log($"[BCInference] Koniec. decyzji={decisionCount}  "
                + $"argmaxPozaStozkiem={argmaxOutsideCone}  wstrzymanych={decisionsHeld}  "
                + $"pozaNavMesh={waypointsOffNavMesh}  skrocone={waypointsShortened}  "
                + $"wBezruchu={decisionsWhileStalled}  "
                + $"manewrow={recoveryCount}");
    }

    void OnDisable() { if (isActive) StopInference(); }
    void OnApplicationQuit() { if (isActive) StopInference(); }
    void OnDestroy() { if (isActive) StopInference(); m_Worker?.Dispose(); }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null)
        {
            if (kb.iKey.wasPressedThisFrame && !isActive) StartInference();
            if (kb.oKey.wasPressedThisFrame && isActive) StopInference();
        }
        if (!isActive) return;

        m_EpisodeTime += Time.deltaTime;
        if (m_CooldownTimer > 0f) m_CooldownTimer -= Time.deltaTime;

        if (isRecovering)
        {
            m_RecoveryTimer -= Time.deltaTime;
            if (chassis != null)
            {
                chassis.SetSpeed(reverseSpeed);
                chassis.SetSteering(m_RecoverySteer);
            }
            if (m_RecoveryTimer <= 0f) EndRecovery();
            return;                     // model nie decyduje w trakcie manewru
        }

        m_DecisionTimer += Time.deltaTime;
        int guard = 0;
        while (m_DecisionTimer >= decisionInterval && guard++ < 4)
        {
            m_DecisionTimer -= decisionInterval;
            MakeDecision();
            if (isRecovering) break;
        }
    }

    // ------------------------------------------------------------------

    /// <summary>Surowy wektor stanu w kolejnosci build_bc_dataset.py:
    /// [posX, posZ] jesli includePosition, yaw, telemetria po filtrach,
    /// scan_dist, scan_age, scan_pitch / scanPitchScaleDeg.</summary>
    private float[] GetCurrentStateVector()
    {
        float[] telemetry = chassis.GetTelemetryState();
        float[] dist = tofScanBuffer != null ? tofScanBuffer.GetNormalizedDistances() : null;
        float[] age = tofScanBuffer != null ? tofScanBuffer.GetNormalizedAges() : null;
        float[] pitch = tofScanBuffer != null ? tofScanBuffer.GetMeasurementPitchesDegrees() : null;

        float[] s = new float[stateDim];
        int k = 0;

        if (addDirectionChannel) s[k++] = Mathf.Sign(drivingDirection);

        if (includePosition)
        {
            s[k++] = carTransform.position.x;
            s[k++] = carTransform.position.z;
        }
        s[k++] = carTransform.eulerAngles.y;

        for (int i = 0; i < telemetry.Length && k < stateDim; i++)
        {
            if (excludePolicyOutputs && i < 4) continue;
            if (i >= yoloFirstIndex && !KeepYoloColumn(i)) continue;
            s[k++] = telemetry[i];
        }

        if (includeScan && dist != null && age != null)
        {
            for (int i = 0; i < dist.Length && k < stateDim; i++) s[k++] = dist[i];
            for (int i = 0; i < age.Length && k < stateDim; i++) s[k++] = age[i];
            if (includeScanPitch && pitch != null)
                for (int i = 0; i < pitch.Length && k < stateDim; i++)
                    s[k++] = pitch[i] / scanPitchScaleDeg;
        }

        if (k != stateDim && !m_StateDimWarned)
        {
            m_StateDimWarned = true;
            Debug.LogError($"[BCInference] Zlozono {k} wartosci, a stateDim={stateDim}. "
                + "Model dostaje ZLE dane. Sprawdz flagi wariantu i liczbe sektorow "
                + "TofScanBuffer wzgledem wydruku export_bc_to_onnx.py.", this);
        }
        return s;
    }

    /// <summary>Czy kolumna telemetrii z bloku YOLO wchodzi do stanu. Musi dawac
    /// DOKLADNIE te sama kolejnosc co get_state_columns() w build_bc_dataset.py,
    /// inaczej model dostaje wartosci na zlych pozycjach.</summary>
    private bool KeepYoloColumn(int telemetryIndex)
    {
        int rel = telemetryIndex - yoloFirstIndex;
        int slot = rel / yoloFeaturesPerSlot;
        int off = rel % yoloFeaturesPerSlot;
        if (slot >= yoloMaxSlots) return false;
        if (yoloDropOffsets != null)
            foreach (int d in yoloDropOffsets)
                if (off == d) return false;
        return true;
    }

    private float BinCenterDeg(int i) => (i + 0.5f) * (360f / nDirBins) - 180f;

    /// <summary>Wybor kierunku z rozkladu, ograniczony do wykonalnego stozka.
    /// Zwraca kat w stopniach; coneMass to masa prawdopodobienstwa w stozku.</summary>
    private float PickDirection(float[] probs, out float coneMass, out bool argmaxOutside)
    {
        int rawBest = 0;
        for (int i = 1; i < nDirBins; i++) if (probs[i] > probs[rawBest]) rawBest = i;
        argmaxOutside = Mathf.Abs(BinCenterDeg(rawBest)) > maxCommandAngleDeg;

        coneMass = 0f;
        int best = -1;
        for (int i = 0; i < nDirBins; i++)
        {
            if (Mathf.Abs(BinCenterDeg(i)) > maxCommandAngleDeg) continue;
            coneMass += probs[i];
            if (best < 0 || probs[i] > probs[best]) best = i;
        }
        if (best < 0) { coneMass = 0f; return 0f; }

        if (!useSoftDirection) return BinCenterDeg(best);

        // srednia kolowa po topK kubelkach w stozku - usuwa kwantyzacje kubelkow
        var idx = new List<int>(nDirBins);
        for (int i = 0; i < nDirBins; i++)
            if (Mathf.Abs(BinCenterDeg(i)) <= maxCommandAngleDeg) idx.Add(i);
        idx.Sort((a, b) => probs[b].CompareTo(probs[a]));

        int n = Mathf.Min(softTopK, idx.Count);
        float sx = 0f, sy = 0f, w = 0f;
        for (int j = 0; j < n; j++)
        {
            float p = probs[idx[j]];
            float a = BinCenterDeg(idx[j]) * Mathf.Deg2Rad;
            sx += p * Mathf.Sin(a);
            sy += p * Mathf.Cos(a);
            w += p;
        }
        if (w < 1e-6f) return BinCenterDeg(best);
        return Mathf.Atan2(sx, sy) * Mathf.Rad2Deg;
    }

    private void BeginRecovery()
    {
        isRecovering = true;
        recoveryCount++;
        m_RecoveryTimer = reverseSeconds;
        m_StalledStreak = 0;

        // skrecamy PRZECIWNIE do ostatniej komendy: auto cofa sie "odkrecajac"
        // od przeszkody, w ktora wjechalo
        float sign = lastAngleDeg >= 0f ? -1f : 1f;
        m_RecoverySteer = sign * reverseSteer;

        if (carAgent != null) carAgent.externalControl = true;
        Debug.LogWarning($"[BCInference] Zaklinowanie po {stallThresholdDecisions} decyzjach "
            + $"w bezruchu - manewr {recoveryCount}: cofanie {reverseSeconds:F1} s, "
            + $"skret {m_RecoverySteer:+0.00;-0.00}", this);
    }

    private void EndRecovery()
    {
        if (!isRecovering) return;
        isRecovering = false;
        m_RecoveryTimer = 0f;
        m_CooldownTimer = recoveryCooldownSeconds;

        if (chassis != null) { chassis.SetSpeed(0f); chassis.SetSteering(0f); }
        if (carAgent != null) carAgent.externalControl = false;

        m_LastDecisionPos = carTransform != null ? carTransform.position : m_LastDecisionPos;
        m_DecisionTimer = decisionInterval;      // od razu nowa decyzja modelu
    }

    /// <summary>Jak daleko mozna jechac w danym kierunku wedlug profilu ToF.
    ///
    /// To jest przenoszalny odpowiednik NavMesh.Raycast: korzysta WYLACZNIE z
    /// czujnika pokladowego, wiec na robocie zadziala identycznie. Cena jest taka,
    /// ze widzi tylko tam, gdzie wiezyczka zdazyla zmierzyc - przy braku swiezego
    /// pomiaru zwraca false i decyzja przechodzi bez sprawdzenia, dokladnie jak
    /// zachowalby sie robot.
    ///
    /// Bierze MINIMUM z wachlarza sektorow wokol kierunku, bo auto ma szerokosc,
    /// a pojedynczy sektor to waski promien.</summary>
    private bool TofReachMeters(float angleDeg, out float reachM)
    {
        reachM = 0f;
        if (tofScanBuffer == null) return false;

        float lo = tofScanBuffer.minYawDegrees;
        float hi = tofScanBuffer.maxYawDegrees;
        if (hi <= lo) return false;
        if (angleDeg < lo || angleDeg > hi) return false;   // poza zakresem wiezyczki

        float[] dist = tofScanBuffer.GetNormalizedDistances();
        float[] age = tofScanBuffer.GetNormalizedAges();
        float[] pitch = tofScanBuffer.GetMeasurementPitchesDegrees();
        int n = tofScanBuffer.SectorCount;
        if (dist.Length < n || age.Length < n) return false;

        float best = float.MaxValue;
        bool any = false;
        for (int sct = 0; sct < n; sct++)
        {
            float center = Mathf.Lerp(lo, hi, (sct + 0.5f) / n);
            if (Mathf.Abs(center - angleDeg) > tofConeHalfWidthDeg) continue;
            if (age[sct] >= 0.999f) continue;                       // brak swiezego pomiaru
            if (pitch != null && sct < pitch.Length
                && Mathf.Abs(pitch[sct]) > tofMaxAbsPitchDeg) continue;

            any = true;
            best = Mathf.Min(best, dist[sct] * tofMaxRangeM);
        }

        if (!any) return false;
        reachM = best;
        return true;
    }

    private void MakeDecision()
    {
        m_StateHistory.Add(GetCurrentStateVector());
        while (m_StateHistory.Count > contextLength) m_StateHistory.RemoveAt(0);

        int tlen = m_StateHistory.Count;
        int pad = contextLength - tlen;

        var states = new Tensor<float>(new TensorShape(1, contextLength, stateDim));
        var mask = new Tensor<float>(new TensorShape(1, contextLength));

        // Padding po LEWEJ, zera + maska 0 - dokladnie jak get_batch() w treningu.
        for (int i = 0; i < contextLength; i++)
        {
            bool isPad = i < pad;
            mask[0, i] = isPad ? 0f : 1f;
            for (int j = 0; j < stateDim; j++)
                states[0, i, j] = isPad ? 0f : m_StateHistory[i - pad][j];
        }

        m_Worker.SetInput("states", states);
        m_Worker.SetInput("attention_mask", mask);
        m_Worker.Schedule();

        float[] probs = (m_Worker.PeekOutput("dir_probs") as Tensor<float>).DownloadToArray();
        float[] magArr = (m_Worker.PeekOutput("mag_m") as Tensor<float>).DownloadToArray();

        states.Dispose();
        mask.Dispose();

        if (probs.Length < nDirBins || float.IsNaN(magArr[0]) || float.IsNaN(probs[0]))
        {
            Debug.LogError("[BCInference] Model zwrocil NaN albo zla liczbe kubelkow "
                + $"({probs.Length} vs nDirBins={nDirBins}). Zatrzymuje inferencje.", this);
            StopInference();
            return;
        }

        float angleDeg = PickDirection(probs, out float coneMass, out bool outside);
        float magM = Mathf.Clamp(magArr[0], minWaypointDistance, maxWaypointDistance);

        if (outside) argmaxOutsideCone++;
        lastConeMass = coneMass;

        Vector3 nowPos = carTransform.position;
        float moved = Vector3.Distance(new Vector3(nowPos.x, 0f, nowPos.z),
                                       new Vector3(m_LastDecisionPos.x, 0f, m_LastDecisionPos.z));
        m_LastDecisionPos = nowPos;

        bool stalled = decisionCount > 0 && moved < stallDistanceM;
        if (stalled) decisionsWhileStalled++;

        if (recoveryEnabled && m_CooldownTimer <= 0f)
        {
            m_StalledStreak = stalled ? m_StalledStreak + 1 : 0;
            if (m_StalledStreak >= stallThresholdDecisions)
            {
                BeginRecovery();
                return;
            }
        }

        bool held = coneMass < minConeMass;
        if (held)
        {
            decisionsHeld++;
            LogDecision(nowPos, angleDeg, magM, coneMass, moved, false, true);
            decisionCount++;
            if (flushEveryDecisions > 0 && decisionCount % flushEveryDecisions == 0) FlushLog();
            return;                     // zostawiamy poprzedni waypoint
        }

        if (straightDeadzoneDeg > 0f && Mathf.Abs(angleDeg) < straightDeadzoneDeg)
            angleDeg = 0f;

        if (debugForceForward) { angleDeg = 0f; magM = 1.5f; }

        // --- limit zasiegu w wybranym kierunku ---
        // Liczony PRZED zbudowaniem punktu, zeby oba tryby dzialaly tak samo:
        // ograniczamy dlugosc komendy, nie przesuwamy gotowego punktu.
        float reachLimit = float.MaxValue;
        Vector3 carPos = carTransform.position;
        Quaternion carYaw = Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f);

        if (traversabilityMode == TraversabilityMode.NavMesh)
        {
            Vector3 from = carPos;
            if (NavMesh.SamplePosition(from, out NavMeshHit fh, 1f, NavMesh.AllAreas))
                from = fh.position;

            float r0 = angleDeg * Mathf.Deg2Rad;
            Vector3 probe = from + carYaw * new Vector3(magM * Mathf.Sin(r0), 0f,
                                                        magM * Mathf.Cos(r0));
            // NavMesh.Raycast zwraca true, gdy trafi w KRAWEDZ siatki
            if (NavMesh.Raycast(from, probe, out NavMeshHit block, NavMesh.AllAreas))
                reachLimit = Vector3.Distance(from, block.position) - traversabilityMargin;
        }
        else if (traversabilityMode == TraversabilityMode.ToF)
        {
            if (TofReachMeters(angleDeg, out float tofReach))
                reachLimit = tofReach - traversabilityMargin;
            else
                tofNoMeasurement++;
        }

        if (magM > reachLimit)
        {
            waypointsShortened++;
            if (reachLimit < minWaypointDistance)
            {
                // nie ma gdzie jechac w tym kierunku - nie ruszamy waypointa
                decisionsHeld++;
                LogDecision(nowPos, angleDeg, magM, coneMass, moved, false, true);
                decisionCount++;
                if (flushEveryDecisions > 0 && decisionCount % flushEveryDecisions == 0)
                    FlushLog();
                return;
            }
            magM = reachLimit;
        }

        float rad = angleDeg * Mathf.Deg2Rad;
        Vector2 local = new Vector2(magM * Mathf.Sin(rad), magM * Mathf.Cos(rad));
        lastLocalAction = local;
        lastAngleDeg = angleDeg;
        lastMagnitudeM = magM;

        Vector3 desired = carPos + carYaw * new Vector3(local.x, 0f, local.y);
        bool offMesh = false;

        if (projectOntoNavMesh)
        {
            if (NavMesh.SamplePosition(desired, out NavMeshHit hit,
                                       navMeshSampleRadius, NavMesh.AllAreas))
                desired = hit.position;
            else { offMesh = true; waypointsOffNavMesh++; }
        }

        if (!offMesh) target.position = desired + new Vector3(0f, 0.05f, 0f);

        LogDecision(nowPos, angleDeg, magM, coneMass, moved, offMesh, false);
        decisionCount++;
        if (flushEveryDecisions > 0 && decisionCount % flushEveryDecisions == 0) FlushLog();
    }

    // ------------------------------------------------------------------

    private void OpenLog()
    {
        m_Csv.Clear();
        m_LogPath = null;
        if (!logDecisions) return;

        string dir = Path.Combine(Application.persistentDataPath, decisionLogFolder);
        Directory.CreateDirectory(dir);
        string tag = string.IsNullOrEmpty(decisionLogTag) ? "run" : Sanitize(decisionLogTag);
        m_LogPath = Path.Combine(dir,
            $"decisions_{tag}_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv");
        File.WriteAllText(m_LogPath,
            "decision,t,posX,posZ,yaw,angleDeg,magM,coneMass,movedSinceLast,offNavMesh,held\n");
        Debug.Log($"[BCInference] Log decyzji -> {m_LogPath}");
    }

    private static string Sanitize(string tag)
    {
        foreach (char c in Path.GetInvalidFileNameChars()) tag = tag.Replace(c, '_');
        return tag;
    }

    private void FlushLog()
    {
        if (m_LogPath != null && m_Csv.Length > 0)
        {
            File.AppendAllText(m_LogPath, m_Csv.ToString());
            m_Csv.Clear();
        }
    }

    private void LogDecision(Vector3 pos, float angleDeg, float magM, float coneMass,
                             float moved, bool offMesh, bool held)
    {
        if (m_LogPath == null) return;
        var ci = CultureInfo.InvariantCulture;
        m_Csv.Append(decisionCount.ToString(ci)).Append(',')
             .Append(m_EpisodeTime.ToString("F2", ci)).Append(',')
             .Append(pos.x.ToString("F3", ci)).Append(',')
             .Append(pos.z.ToString("F3", ci)).Append(',')
             .Append(carTransform.eulerAngles.y.ToString("F2", ci)).Append(',')
             .Append(angleDeg.ToString("F2", ci)).Append(',')
             .Append(magM.ToString("F3", ci)).Append(',')
             .Append(coneMass.ToString("F4", ci)).Append(',')
             .Append(moved.ToString("F3", ci)).Append(',')
             .Append(offMesh ? "1" : "0").Append(',')
             .Append(held ? "1" : "0")
             .AppendLine();
    }

    void OnDrawGizmos()
    {
        if (!drawGizmos || !isActive || carTransform == null || target == null) return;

        if (isRecovering)
        {
            Gizmos.color = Color.magenta;      // magenta = manewr wychodzenia
            Gizmos.DrawRay(carTransform.position, -carTransform.forward * 1.5f);
            Gizmos.DrawWireSphere(carTransform.position, 0.5f);
            return;
        }

        Gizmos.color = lastConeMass < minConeMass ? Color.red : Color.green;
        Gizmos.DrawLine(carTransform.position, target.position);
        Gizmos.DrawWireSphere(target.position, 0.25f);

        // granice wykonalnego stozka
        Gizmos.color = new Color(1f, 1f, 0f, 0.4f);
        var yaw = Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f);
        foreach (float sgn in new[] { -1f, 1f })
            Gizmos.DrawRay(carTransform.position,
                yaw * Quaternion.Euler(0f, sgn * maxCommandAngleDeg, 0f)
                    * Vector3.forward * maxWaypointDistance);
    }
}