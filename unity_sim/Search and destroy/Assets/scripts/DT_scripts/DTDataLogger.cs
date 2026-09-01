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
    public float[] telemetry;   // 17 wartosci z Chassis.GetTelemetryState()
    public bool collision;
    public float expertX;       // ETYKIETA: wektor do pursuit pointa w ukladzie auta
    public float expertZ;
    public bool expertValid;    // czy AutoExplorer mial w tym kroku wyznaczona trase
}

public class DTDataLogger : MonoBehaviour
{
    [Header("References")]
    public Chassis chassis;
    public Transform carTransform;
    public AutoExplorer autoExplorer;
    public CarAgent carAgent;

    [Header("Recording Settings")]
    public float logIntervalSeconds = 0.1f;   // 10 Hz
    public string outputFolder = "DTDataset";
    [Tooltip("Unikalny prefiks dla tej instancji (np. przy kilku rownoleglych arenach).")]
    public string instancePrefix = "";

    [Header("Auto-Chunking")]
    [Tooltip("Po ilu krokach automatycznie zapisac fragment jako osobny epizod i kontynuowac. 0 = wylaczone.")]
    public int autoEndAfterSteps = 800;

    [Header("Restart przy zaklinowaniu")]
    [Tooltip("Fragmenty krotsze niz tyle krokow sa ODRZUCANE zamiast zapisywane. Przy decymacji 15x to 150 krokow = 10 decyzji modelu - ponizej tego fragment nie ma wartosci jako trajektoria. Uwaga: filtrowanie tutaj systematycznie usuwa POCZATKI trajektorii, wiec nie ustawiaj tego wysoko.")]
    public int minEpisodeSteps = 150;

    [Header("Runtime State (read-only)")]
    public bool isRecording = false;
    public int currentEpisodeId = 0;
    public int stepsInCurrentChunk = 0;
    public int savedEpisodes = 0;
    public int discardedFragments = 0;

    private List<DTStepData> buffer = new List<DTStepData>();
    private float timer = 0f;
    private int stepCounter = 0;

    void Reset()
    {
        chassis = GetComponent<Chassis>();
        carTransform = transform;
        carAgent = GetComponent<CarAgent>();
    }

    void Awake()
    {
        // Bez tego kazde wejscie w Play Mode zaczynaloby numeracje od 0
        // i NADPISYWALO pliki z poprzednich sesji.
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

    void FixedUpdate()
    {
        if (!isRecording) return;

        timer += Time.fixedDeltaTime;
        if (timer < logIntervalSeconds) return;
        timer = 0f;

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

        // Etykieta prosto od eksperta, zamiast relabelingu z przyszlej pozycji
        // auta. Dzieki temu nie zaleza od tego, jak poradzil sobie driver -
        // szarpanie sie i cofanie przestaje zanieczyszczac dane.
        Vector2 expert = Vector2.zero;
        bool expertOk = false;
        if (autoExplorer != null && autoExplorer.isExploring)
        {
            expert = autoExplorer.expertLocalWaypoint;
            expertOk = true;
        }

        buffer.Add(new DTStepData
        {
            t = stepCounter++,
            posX = carTransform.position.x,
            posZ = carTransform.position.z,
            yaw = carTransform.eulerAngles.y,
            telemetry = chassis.GetTelemetryState(),
            collision = collisionFlag,
            expertX = expert.x,
            expertZ = expert.y,
            expertValid = expertOk,
        });
        stepsInCurrentChunk = buffer.Count;
    }

    // ---- sterowanie sesja -------------------------------------------------

    public void StartEpisode()
    {
        buffer.Clear();
        stepCounter = 0;
        timer = 0f;
        stepsInCurrentChunk = 0;
        isRecording = true;

        if (autoExplorer != null) autoExplorer.StartExploring();

        Debug.Log($"[DTDataLogger] Start sesji (epizod {currentEpisodeId}), "
                + $"auto-chunk co {autoEndAfterSteps} krokow");
    }

    private void SaveCurrentChunkAndContinue()
    {
        SaveEpisodeToCsv();
        currentEpisodeId++;
        savedEpisodes++;

        buffer.Clear();
        stepCounter = 0;
        stepsInCurrentChunk = 0;
        // NIE ruszamy timer/isRecording/autoExplorer - jazda trwa nieprzerwanie
    }

    /// <summary>
    /// Zamyka biezacy fragment i pozwala kontynuowac nagrywanie od nowa.
    /// Wolane przez AutoExplorer po wykryciu zaklinowania, ZANIM auto zostanie
    /// przeniesione - dzieki temu respawn nie trafia do srodka trajektorii,
    /// tylko rozpoczyna nowy epizod.
    ///
    /// Fragment krotszy niz minEpisodeSteps jest odrzucany: koncowka przed
    /// zaklinowaniem to zapis nieudanych prob wyjscia, czyli dokladnie to,
    /// czego nie chcemy w etykietach.
    /// </summary>
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
        stepsInCurrentChunk = 0;
        timer = 0f;
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

    // ---- zapis ------------------------------------------------------------

    private void SaveEpisodeToCsv()
    {
        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"{FilePrefix()}{currentEpisodeId:D4}.csv");

        var sb = new StringBuilder();
        sb.Append("t,posX,posZ,yaw");
        for (int i = 0; i < 17; i++) sb.Append($",telem_{i}");
        sb.Append(",collision,expert_x,expert_z,expert_valid").AppendLine();

        var inv = CultureInfo.InvariantCulture;
        foreach (var s in buffer)
        {
            sb.Append(s.t.ToString(inv));
            sb.Append(',').Append(s.posX.ToString(inv));
            sb.Append(',').Append(s.posZ.ToString(inv));
            sb.Append(',').Append(s.yaw.ToString(inv));
            for (int i = 0; i < s.telemetry.Length; i++)
                sb.Append(',').Append(s.telemetry[i].ToString(inv));
            sb.Append(',').Append(s.collision ? "1" : "0");
            sb.Append(',').Append(s.expertX.ToString(inv));
            sb.Append(',').Append(s.expertZ.ToString(inv));
            sb.Append(',').Append(s.expertValid ? "1" : "0");
            sb.AppendLine();
        }

        File.WriteAllText(path, sb.ToString());
        Debug.Log($"[DTDataLogger] Zapisano epizod {currentEpisodeId} ({buffer.Count} krokow)");
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