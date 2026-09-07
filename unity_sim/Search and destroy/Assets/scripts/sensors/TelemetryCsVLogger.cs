using UnityEngine;
using System.IO;
using System;
using System.Text;
using System.Globalization;

public class TelemetryCsVLogger : MonoBehaviour
{
    [Header("References")]
    public Chassis chassis;
    [Header("Log Settings")]
    public string fileName = "SimulationData.csv";
    public bool logOnFixedUpdate = true;

    private StreamWriter m_Writer;
    private string m_FilePath;

    void Start()
    {
        if (chassis == null) chassis = GetComponent<Chassis>();

        m_FilePath = Path.Combine(Application.dataPath, "../../../logi/sim", fileName);
        m_Writer = new StreamWriter(m_FilePath, false);

        StringBuilder header = new StringBuilder();
        header.Append("Timestamp,Motor,Steering,CamPitch,CamYaw,AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ,ToF");

        m_Writer.WriteLine(header.ToString());
        Debug.Log($"[TelemetryLogger] Started logging CSV to: {m_FilePath}");
    }

    void FixedUpdate()
    {
        if (logOnFixedUpdate)
        {
            LogCurrentState();
        }
    }

    public void LogCurrentState()
    {
        if(m_Writer == null) return;

        float[] telemetry = chassis.GetTelemetryState();
        StringBuilder sb = new StringBuilder();
        string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
        sb.Append($"[{timestamp}]");

        for (int i = 0; i < telemetry.Length; i++)
        {
            if (i < 11)
            {
                sb.Append("," + telemetry[i].ToString("F2", CultureInfo.InvariantCulture));
            }
        }
        m_Writer.WriteLine(sb.ToString());
    }

    void OnDestroy()
    {
        if (m_Writer != null)
        {
            m_Writer.Flush();
            m_Writer.Close();
            Debug.Log("[TelemetryLogger] CSV Log saved sucesfully");
        }
    }
}
