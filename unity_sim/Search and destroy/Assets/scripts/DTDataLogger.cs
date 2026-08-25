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
}

public class DTDataLogger : MonoBehaviour
{
    [Header("References")]
    public Chassis chassis;
    public Transform carTransform;
    public AutoExplorer autoExplorer;

    [Header("Recording Settings")]
    public float logIntervalSeconds = 0.1f; // 10 Hz
    public string outputFolder = "DTDataset";

    [Header("Auto-Chunking (automatyczny podział na epizody)")]
    [Tooltip("Po ilu zalogowanych krokach automatycznie zapisac fragment jako osobny epizod, 0 = wylaczone (tylko reczne R/T/Y).")]
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
    }

    void Awake()
    {
        currentEpisodeId = GetNextAvailableEpisodeId();
    }

    private int GetNextAvailableEpisodeId()
    {
        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        if (!Directory.Exists(dir)) return 0;

        int maxId = -1;
        string[] files = Directory.GetFiles(dir, "episode_*.csv");
        foreach (string f in files)
        {
            string nameOnly = Path.GetFileNameWithoutExtension(f);
            string numberPart = nameOnly.Substring("episode_".Length);
            if (int.TryParse(numberPart, out int id))
            {
                if (id > maxId) maxId = id;
            }
        }

        int nextId = maxId + 1;
        Debug.Log($"[DTDataLogger] Znaleziono istniejace pliki w {dir}, "
                   + $"kontynuuje numeracje od episode_{nextId:D4}.csv");
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
        {
            SaveCurrentChunkAndContinue();
        }
    }

    private void LogStep()
    {
        if (chassis == null || carTransform == null) return;

        var step = new DTStepData
        {
            t = stepCounter++,
            posX = carTransform.position.x,
            posZ = carTransform.position.z,
            yaw = carTransform.eulerAngles.y,
            telemetry = chassis.GetTelemetryState() // 17 wartości
        };

        buffer.Add(step);
        stepsInCurrentChunk = buffer.Count;
    }

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

    private void SaveCurrentChunkAndContinue()
    {
        SaveEpisodeToCsv();
        currentEpisodeId++;

        buffer.Clear();
        stepCounter = 0;
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
        string path = Path.Combine(dir, $"episode_{currentEpisodeId:D4}.csv");

        var sb = new StringBuilder();

        sb.Append("t,posX,posZ,yaw");
        for (int i = 0; i < 17; i++) sb.Append($",telem_{i}");
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