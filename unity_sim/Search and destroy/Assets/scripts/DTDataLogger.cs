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
    public float[] telemetry; // 17 wartości z Chassis.GetTelemetryState()
    public bool collision;    // NOWOSC: czy w tym kroku wystapila realna kolizja (z CarAgent)
}

public class DTDataLogger : MonoBehaviour
{
    [Header("References")]
    public Chassis chassis;
    public Transform carTransform;
    public AutoExplorer autoExplorer; // opcjonalne - jeśli chcesz automatyczną eksplorację
    public CarAgent carAgent; // NOWOSC: potrzebne do odczytu flagi kolizji

    [Header("Recording Settings")]
    public float logIntervalSeconds = 0.1f; // 10 Hz
    public string outputFolder = "DTDataset";
    [Tooltip("Unikalny prefiks dla tej instancji (np. przy kilku rownoleglych arenach/autach), zeby pliki CSV z roznych aren sie nie nadpisywaly. Zostaw puste dla pojedynczego auta.")]
    public string instancePrefix = "";

    [Header("Auto-Chunking (automatyczne krojenie na epizody)")]
    [Tooltip("Po ilu zalogowanych krokach automatycznie zapisac fragment jako osobny epizod i kontynuowac dalej. 0 = wylaczone (tylko reczne R/T/Y).")]
    public int autoEndAfterSteps = 800;

    [Header("Runtime State (read-only)")]
    public bool isRecording = false;
    public int currentEpisodeId = 0;
    public int stepsInCurrentChunk = 0;

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
        // Wazne: bez tego, kazde ponowne wejscie w Play Mode zaczynaloby
        // numeracje epizodow od 0, NADPISUJAC pliki z poprzednich sesji
        // nagrywania o tych samych numerach!
        currentEpisodeId = GetNextAvailableEpisodeId();
    }

    private string FilePrefix()
    {
        return string.IsNullOrEmpty(instancePrefix) ? "episode_" : $"{instancePrefix}_episode_";
    }

    private int GetNextAvailableEpisodeId()
    {
        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        if (!Directory.Exists(dir)) return 0;

        string prefix = FilePrefix();
        int maxId = -1;
        string[] files = Directory.GetFiles(dir, $"{prefix}*.csv");
        foreach (string f in files)
        {
            string nameOnly = Path.GetFileNameWithoutExtension(f); // np. "area1_episode_0007" albo "episode_0007"
            string numberPart = nameOnly.Substring(prefix.Length);
            if (int.TryParse(numberPart, out int id))
            {
                if (id > maxId) maxId = id;
            }
        }

        int nextId = maxId + 1;
        Debug.Log($"[DTDataLogger:{instancePrefix}] Znaleziono istniejace pliki w {dir}, "
                   + $"kontynuuje numeracje od {prefix}{nextId:D4}.csv");
        return nextId;
    }

    void FixedUpdate()
    {
        if (!isRecording) return;

        timer += Time.fixedDeltaTime;
        if (timer < logIntervalSeconds) return;
        timer = 0f;

        LogStep();

        // Auto-chunking: co autoEndAfterSteps krokow zapisujemy fragment
        // i kontynuujemy dalej BEZ przerywania jazdy/eksploracji.
        if (autoEndAfterSteps > 0 && buffer.Count >= autoEndAfterSteps)
        {
            SaveCurrentChunkAndContinue();
        }
    }

    private void LogStep()
    {
        if (chassis == null || carTransform == null) return;

        bool collisionFlag = false;
        if (carAgent != null)
        {
            collisionFlag = carAgent.hadCollisionThisStep;
            carAgent.hadCollisionThisStep = false; // reset po odczycie, zeby nie "przecieklo" do kolejnego kroku
        }

        var step = new DTStepData
        {
            t = stepCounter++,
            posX = carTransform.position.x,
            posZ = carTransform.position.z,
            yaw = carTransform.eulerAngles.y,
            telemetry = chassis.GetTelemetryState(), // 17 wartości
            collision = collisionFlag
        };

        buffer.Add(step);
        stepsInCurrentChunk = buffer.Count;
    }

    // Wywołuj żeby rozpocząć CAŁĄ sesję nagrywania (recznie, klawisz R)
    public void StartEpisode()
    {
        buffer.Clear();
        stepCounter = 0;
        timer = 0f;
        stepsInCurrentChunk = 0;
        isRecording = true;

        if (autoExplorer != null) autoExplorer.StartExploring();

        Debug.Log($"[DTDataLogger] Started recording session (episode {currentEpisodeId}), "
                   + $"auto-chunk co {autoEndAfterSteps} krokow");
    }

    // Automatyczne "krojenie": zapisz obecny fragment jako plik, zresetuj bufor,
    // ale NIE zatrzymuj nagrywania ani eksploracji - kontynuuj dalej bez przerwy.
    private void SaveCurrentChunkAndContinue()
    {
        SaveEpisodeToCsv();
        currentEpisodeId++;

        buffer.Clear();
        stepCounter = 0;
        stepsInCurrentChunk = 0;
        // uwaga: NIE resetujemy timer ani isRecording, NIE ruszamy autoExplorer -
        // jazda i eksploracja trwaja nieprzerwanie, tylko zaczynamy nowy plik
    }

    // Wywołuj żeby zakończyć CAŁĄ sesję (recznie, klawisz T lub Y)
    public void EndEpisode(bool discard = false)
    {
        isRecording = false;

        if (autoExplorer != null) autoExplorer.StopExploring();

        if (!discard && buffer.Count > 0)
        {
            SaveEpisodeToCsv();
            currentEpisodeId++;
        }
        else if (discard)
        {
            Debug.Log($"[DTDataLogger] Odrzucono ostatni, niedokonczony fragment ({buffer.Count} krokow)");
        }

        buffer.Clear();
        stepsInCurrentChunk = 0;
    }

    private void SaveEpisodeToCsv()
    {
        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"{FilePrefix()}{currentEpisodeId:D4}.csv");

        var sb = new StringBuilder();

        sb.Append("t,posX,posZ,yaw");
        for (int i = 0; i < 17; i++) sb.Append($",telem_{i}");
        sb.Append(",collision");
        sb.AppendLine();

        foreach (var s in buffer)
        {
            sb.Append(s.t.ToString(CultureInfo.InvariantCulture));
            sb.Append(",").Append(s.posX.ToString(CultureInfo.InvariantCulture));
            sb.Append(",").Append(s.posZ.ToString(CultureInfo.InvariantCulture));
            sb.Append(",").Append(s.yaw.ToString(CultureInfo.InvariantCulture));

            for (int i = 0; i < s.telemetry.Length; i++)
            {
                sb.Append(",").Append(s.telemetry[i].ToString(CultureInfo.InvariantCulture));
            }

            sb.Append(",").Append(s.collision ? "1" : "0");

            sb.AppendLine();
        }

        File.WriteAllText(path, sb.ToString());
        Debug.Log($"[DTDataLogger] Saved episode {currentEpisodeId} ({buffer.Count} steps) to {path}");
    }

    void Update()
    {
        var keyboard = Keyboard.current;
        if (keyboard == null) return;

        if (keyboard.rKey.wasPressedThisFrame && !isRecording) StartEpisode();
        if (keyboard.tKey.wasPressedThisFrame && isRecording) EndEpisode(discard: false);
        if (keyboard.yKey.wasPressedThisFrame && isRecording) EndEpisode(discard: true);
    }
}