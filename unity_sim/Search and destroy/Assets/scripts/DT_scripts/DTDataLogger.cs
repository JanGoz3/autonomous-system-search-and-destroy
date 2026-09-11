using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

[System.Serializable]
public struct DTStepData
{
    public int t;
    public float posX;
    public float posZ;
    public float yaw;
    public float[] telemetry;
    public bool collision;
    public float expertX;       // ETYKIETA: wektor do pursuit pointa w ukladzie auta
    public float expertZ;
    public bool expertValid;    // czy etykieta nadaje sie do lossu (ExpertLabelUsable)
    public float progressAlongRoute;   // rzut auta na trase, w metrach po luku
    public float deviationFromRoute;   // odleglosc auta od trasy, w metrach
    public float turretPitchDeg;
    public float turretYawDeg;
    public float[] scanDistances;
    public float[] scanAges;
    public float[] scanPitches;
}

public class DTDataLogger : MonoBehaviour
{
    [Header("References")]
    public Chassis chassis;
    public ServosCamera servosCamera;
    public TofScanBuffer tofScanBuffer;
    public Transform carTransform;
    public AutoExplorer autoExplorer;
    public CarAgent carAgent;

    [Header("Recording Settings")]
    [Tooltip("Docelowy odstep miedzy krokami logu. Zaokraglany do calkowitej liczby krokow "
           + "fizyki, wiec faktyczna czestotliwosc to Time.fixedDeltaTime * stride. "
           + "Przy fixedDeltaTime = 0.02 i 0.1 s otrzymujesz dokladnie 10 Hz.")]
    public float logIntervalSeconds = 0.1f;   // 10 Hz
    public string outputFolder = "DTDataset";
    [Tooltip("Unikalny prefiks dla tej instancji (np. przy kilku rownoleglych arenach).")]
    public string instancePrefix = "";

    [Header("Auto-Chunking")]
    [Tooltip("Po ilu krokach automatycznie zapisac fragment jako osobny epizod i kontynuowac. 0 = wylaczone.")]
    public int autoEndAfterSteps = 800;

    [Header("Restart przy zaklinowaniu")]
    [Tooltip("Fragmenty krotsze niz tyle krokow sa ODRZUCANE zamiast zapisywane. Przy decymacji 10x to 150 krokow = 15 decyzji modelu - ponizej tego fragment nie ma wartosci jako trajektoria. Uwaga: filtrowanie tutaj systematycznie usuwa POCZATKI trajektorii, wiec nie ustawiaj tego wysoko.")]
    public int minEpisodeSteps = 200;

    [Header("Runtime State (read-only)")]
    public bool isRecording = false;
    public int currentEpisodeId = 0;
    public int stepsInCurrentChunk = 0;
    public int savedEpisodes = 0;
    public int discardedFragments = 0;
    [Tooltip("Ile krokow fizyki przypada na jeden krok logu. Wyliczane przy starcie sesji.")]
    public int logStride = 5;

    private List<DTStepData> buffer = new List<DTStepData>();
    private int fixedStepCounter = 0;
    private int stepCounter = 0;
    private int scanSectorsThisChunk = 0;

    void Reset()
    {
        chassis = GetComponent<Chassis>();
        servosCamera = GetComponentInChildren<ServosCamera>(true);
        tofScanBuffer = GetComponentInChildren<TofScanBuffer>(true);
        carTransform = transform;
        carAgent = GetComponent<CarAgent>();
        autoExplorer = GetComponent<AutoExplorer>();
    }

    void Awake()
    {
        currentEpisodeId = GetNextAvailableEpisodeId();
    }

    private string FilePrefix() =>
        string.IsNullOrEmpty(instancePrefix) ? "episode_" : $"{instancePrefix}_episode_";

    private int GetNextAvailableEpisodeId()
    {
        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        if (!Directory.Exists(dir)) return 0;

        string prefix = FilePrefix();
        int maxId = -1;
        foreach (string f in Directory.GetFiles(dir, $"{prefix}*.csv"))
        {
            string numberPart = Path.GetFileNameWithoutExtension(f).Substring(prefix.Length);
            if (int.TryParse(numberPart, out int id) && id > maxId) maxId = id;
        }

        int nextId = maxId + 1;
        Debug.Log($"[DTDataLogger:{instancePrefix}] Kontynuuje numeracje od {prefix}{nextId:D4}.csv");
        return nextId;
    }

    private int ComputeStride()
    {
        int stride = Mathf.Max(1, Mathf.RoundToInt(logIntervalSeconds / Time.fixedDeltaTime));
        float actual = stride * Time.fixedDeltaTime;
        if (Mathf.Abs(actual - logIntervalSeconds) > 1e-4f)
            Debug.LogWarning($"[DTDataLogger] logIntervalSeconds = {logIntervalSeconds} nie jest "
                + $"wielokrotnoscia Time.fixedDeltaTime = {Time.fixedDeltaTime}. Faktyczny odstep "
                + $"to {actual:F4} s ({1f / actual:F2} Hz). Uzgodnij to z DECIMATE i "
                + "rewardTickInterval w DTInference, inaczej skala czasu w danych i w inferencji "
                + "sie rozjedzie.", this);
        return stride;
    }

    void FixedUpdate()
    {
        if (!isRecording) return;
        if (fixedStepCounter++ % logStride != 0) return;

        LogStep();

        if (autoEndAfterSteps > 0 && buffer.Count >= autoEndAfterSteps)
            SaveCurrentChunkAndContinue();
    }

    private void LogStep()
    {
        if (chassis == null || carTransform == null) return;

        bool collisionFlag = false;
        if (carAgent != null)
        {
            collisionFlag = carAgent.hadCollisionThisStep;
            carAgent.hadCollisionThisStep = false;
        }

        Vector2 expert = Vector2.zero;
        bool expertOk = false;
        float progress = 0f, deviation = 0f;
        if (autoExplorer != null && autoExplorer.isExploring)
        {
            expert = autoExplorer.expertLocalWaypoint;

            // BYLO: expertOk = true bezwarunkowo. Kolumna expert_valid byla wiec
            // stale rowna 1 i filtr etykiet fallbackowych nie dzialal nigdy.
            // ExpertLabelUsable sprawdza pelna sciezke NavMesh, niezerowa dlugosc
            // ORAZ |kat| <= maxLabelAngleDeg.
            expertOk = autoExplorer.ExpertLabelUsable;

            progress = autoExplorer.progressAlongRoute;
            deviation = autoExplorer.deviationFromRoute;
        }

        float[] telemetrySnapshot = (float[])chassis.GetTelemetryState().Clone();

        (float pitch, float yaw) turretAngles = servosCamera != null
            ? servosCamera.GetActualPitchYawDegrees()
            : (0f, 0f);

        buffer.Add(new DTStepData
        {
            t = stepCounter++,
            posX = carTransform.position.x,
            posZ = carTransform.position.z,
            yaw = carTransform.eulerAngles.y,
            telemetry = telemetrySnapshot,
            collision = collisionFlag,
            expertX = expert.x,
            expertZ = expert.y,
            expertValid = expertOk,
            progressAlongRoute = progress,
            deviationFromRoute = deviation,
            turretPitchDeg = turretAngles.pitch,
            turretYawDeg = turretAngles.yaw,
            scanDistances = tofScanBuffer != null ? tofScanBuffer.GetNormalizedDistances() : null,
            scanAges = tofScanBuffer != null ? tofScanBuffer.GetNormalizedAges() : null,
            scanPitches = tofScanBuffer != null ? tofScanBuffer.GetMeasurementPitchesDegrees() : null,
        });
        stepsInCurrentChunk = buffer.Count;
    }


    public void StartEpisode()
    {
        if (servosCamera == null)
            Debug.LogWarning("[DTDataLogger] Brak referencji ServosCamera: turret_pitch_deg "
                + "i turret_yaw_deg beda zapisywane jako 0. Podepnij ServosCamera w Inspectorze.", this);

        if (tofScanBuffer == null)
        {
            Debug.LogWarning("[DTDataLogger] Brak referencji TofScanBuffer: kolumny scan_* "
                + "beda pominiete. Podepnij bufor w Inspectorze.", this);
            scanSectorsThisChunk = 0;
        }
        else
        {
            tofScanBuffer.Clear();
            scanSectorsThisChunk = tofScanBuffer.SectorCount;
        }

        logStride = ComputeStride();

        buffer.Clear();
        stepCounter = 0;
        fixedStepCounter = 0;
        stepsInCurrentChunk = 0;
        isRecording = true;

        if (autoExplorer != null) autoExplorer.StartExploring();

        Debug.Log($"[DTDataLogger] Start sesji (epizod {currentEpisodeId}), "
                + $"co {logStride} krokow fizyki = {1f / (logStride * Time.fixedDeltaTime):F2} Hz, "
                + $"auto-chunk co {autoEndAfterSteps} krokow, sektorow skanu: {scanSectorsThisChunk}");
    }

    private void SaveCurrentChunkAndContinue()
    {
        SaveEpisodeToCsv();
        currentEpisodeId++;
        savedEpisodes++;

        buffer.Clear();
        stepCounter = 0;
        fixedStepCounter = 0;
        stepsInCurrentChunk = 0;
    }

    public void RestartEpisode(string reason)
    {
        if (!isRecording) return;

        if (buffer.Count >= minEpisodeSteps)
        {
            SaveEpisodeToCsv();
            currentEpisodeId++;
            savedEpisodes++;
            Debug.Log($"[DTDataLogger] Restart ({reason}): zapisano fragment "
                    + $"{buffer.Count} krokow");
        }
        else
        {
            discardedFragments++;
            Debug.Log($"[DTDataLogger] Restart ({reason}): odrzucono fragment "
                    + $"{buffer.Count} krokow (< {minEpisodeSteps})");
        }

        buffer.Clear();
        stepCounter = 0;
        fixedStepCounter = 0;
        stepsInCurrentChunk = 0;
    }

    public void EndEpisode(bool discard = false)
    {
        isRecording = false;
        if (autoExplorer != null) autoExplorer.StopExploring();

        if (!discard && buffer.Count > 0)
        {
            SaveEpisodeToCsv();
            currentEpisodeId++;
            savedEpisodes++;
        }
        else if (discard)
        {
            Debug.Log($"[DTDataLogger] Odrzucono niedokonczony fragment ({buffer.Count} krokow)");
        }

        buffer.Clear();
        stepsInCurrentChunk = 0;
        Debug.Log($"[DTDataLogger] Koniec sesji. Zapisane: {savedEpisodes}, "
                + $"odrzucone fragmenty: {discardedFragments}");
    }

    private static float SafeAt(float[] arr, int i) =>
        arr != null && i < arr.Length ? arr[i] : 0f;

    private void SaveEpisodeToCsv()
    {
        if (buffer.Count == 0) return;

        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"{FilePrefix()}{currentEpisodeId:D4}.csv");

        int telemetryCount = buffer[0].telemetry.Length;
        int scanCount = scanSectorsThisChunk;
        bool mismatch = false;

        foreach (var s in buffer)
        {
            if (s.telemetry.Length != telemetryCount)
            {
                Debug.LogError("[DTDataLogger] Zmienna dlugosc telemetrii w obrebie fragmentu - "
                    + "plik bedzie niespojny.");
                break;
            }
            if ((s.scanDistances?.Length ?? 0) != scanCount) { mismatch = true; }
        }
        if (mismatch)
            Debug.LogWarning($"[DTDataLogger] Liczba sektorow skanu zmienila sie w trakcie "
                + $"fragmentu. Uzywam {scanCount} kolumn, brakujace pola zapisuje jako 0.");

        var sb = new StringBuilder();
        sb.Append("t,posX,posZ,yaw");
        for (int i = 0; i < telemetryCount; i++) sb.Append($",telem_{i}");
        sb.Append(",collision,expert_x,expert_z,expert_valid");
        sb.Append(",progress_along_route,deviation_from_route");
        sb.Append(",turret_pitch_deg,turret_yaw_deg");
        for (int s = 0; s < scanCount; s++) sb.Append($",scan_dist_{s}");
        for (int s = 0; s < scanCount; s++) sb.Append($",scan_age_{s}");
        for (int s = 0; s < scanCount; s++) sb.Append($",scan_pitch_{s}");
        sb.AppendLine();

        var inv = CultureInfo.InvariantCulture;
        foreach (var s in buffer)
        {
            sb.Append(s.t.ToString(inv));
            sb.Append(',').Append(s.posX.ToString(inv));
            sb.Append(',').Append(s.posZ.ToString(inv));
            sb.Append(',').Append(s.yaw.ToString(inv));
            for (int i = 0; i < telemetryCount; i++)
                sb.Append(',').Append(SafeAt(s.telemetry, i).ToString(inv));
            sb.Append(',').Append(s.collision ? "1" : "0");
            sb.Append(',').Append(s.expertX.ToString(inv));
            sb.Append(',').Append(s.expertZ.ToString(inv));
            sb.Append(',').Append(s.expertValid ? "1" : "0");
            sb.Append(',').Append(s.progressAlongRoute.ToString(inv));
            sb.Append(',').Append(s.deviationFromRoute.ToString(inv));
            sb.Append(',').Append(s.turretPitchDeg.ToString(inv));
            sb.Append(',').Append(s.turretYawDeg.ToString(inv));
            for (int i = 0; i < scanCount; i++)
                sb.Append(',').Append(SafeAt(s.scanDistances, i).ToString(inv));
            for (int i = 0; i < scanCount; i++)
                sb.Append(',').Append(SafeAt(s.scanAges, i).ToString(inv));
            for (int i = 0; i < scanCount; i++)
                sb.Append(',').Append(SafeAt(s.scanPitches, i).ToString(inv));
            sb.AppendLine();
        }

        File.WriteAllText(path, sb.ToString());
        Debug.Log($"[DTDataLogger] Zapisano epizod {currentEpisodeId} ({buffer.Count} krokow, "
                + $"{telemetryCount} telem + {scanCount} sektorow skanu)");
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb == null) return;

        if (kb.rKey.wasPressedThisFrame && !isRecording) StartEpisode();
        if (kb.tKey.wasPressedThisFrame && isRecording) EndEpisode(discard: false);
        if (kb.yKey.wasPressedThisFrame && isRecording) EndEpisode(discard: true);
    }
}