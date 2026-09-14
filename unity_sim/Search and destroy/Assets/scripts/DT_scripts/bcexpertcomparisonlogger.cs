using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;


public class BCExpertComparisonLogger : MonoBehaviour
{
    [Header("References")]
    public BCInference bc;
    public AutoExplorer autoExplorer;
    public Chassis chassis;
    public Transform carTransform;

    [Header("Ustawienia")]
    public string outputFolder = "BCExpertComparison";
    public string runTag = "";
    [Tooltip("Zapis na dysk co tyle wierszy.")]
    public int flushEveryRows = 25;
    [Tooltip("Automatycznie wlacza AutoExplorer z driveTarget = false przy starcie "
           + "logowania. Bez respawnu - auto zostaje tam, gdzie jest.")]
    public bool autoConfigureExpert = true;

    [Header("Runtime (read-only)")]
    public bool isLogging = false;
    public int rows = 0;
    [Tooltip("Sredni |bad kata| BC wzgledem eksperta, tylko probki z deviation < 1 m.")]
    public float meanAngleErrorNear = 0f;
    [Tooltip("To samo dla deviation >= 1 m, czyli stanow poza rozkladem treningowym.")]
    public float meanAngleErrorFar = 0f;
    public int nNear = 0;
    public int nFar = 0;

    private int m_LastSerial = -1;
    private readonly StringBuilder m_Csv = new StringBuilder();
    private string m_Path;
    private float m_T;
    private double m_SumNear, m_SumFar;

    void Reset()
    {
        bc = GetComponent<BCInference>();
        autoExplorer = GetComponent<AutoExplorer>();
        chassis = GetComponent<Chassis>();
        carTransform = transform;
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null && kb.lKey.wasPressedThisFrame)
        {
            if (isLogging) StopLogging(); else StartLogging();
        }
        if (isLogging) m_T += Time.deltaTime;
    }

    public void StartLogging()
    {
        if (bc == null || autoExplorer == null || carTransform == null)
        {
            Debug.LogError("[BCExpertComparison] Brak referencji bc/autoExplorer/carTransform.", this);
            return;
        }
        if (!bc.isActive)
            Debug.LogWarning("[BCExpertComparison] BCInference nie jest aktywny - wcisnij I.", this);

        if (autoConfigureExpert)
        {
            autoExplorer.driveTarget = false;
            if (!autoExplorer.isExploring)
            {
                bool prev = autoExplorer.respawnOnStart;
                autoExplorer.respawnOnStart = false;
                autoExplorer.StartExploring();
                autoExplorer.respawnOnStart = prev;
            }
        }
        if (autoExplorer.driveTarget)
            Debug.LogError("[BCExpertComparison] AutoExplorer.driveTarget = true. Oba "
                + "komponenty pisza po target.position - porownanie bedzie bezwartosciowe.", this);

        string dir = Path.Combine(Application.persistentDataPath, outputFolder);
        Directory.CreateDirectory(dir);
        string tag = string.IsNullOrEmpty(runTag) ? "run" : runTag;
        foreach (char c in Path.GetInvalidFileNameChars()) tag = tag.Replace(c, '_');
        m_Path = Path.Combine(dir, $"cmp_{tag}_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv");

        File.WriteAllText(m_Path,
            "decision,t,posX,posZ,yaw,"
          + "bc_raw_angle,bc_angle,bc_mag,"
          + "bc_applied_angle,bc_applied_mag,"
          + "expert_angle,expert_mag,expert_valid,"
          + "angle_error,endpoint_error,"
          + "cone_mass,progress,deviation,speed,moved_since_last,"
          + "held,applied,recovery\n");

        m_Csv.Clear();
        rows = 0; nNear = 0; nFar = 0;
        m_SumNear = 0; m_SumFar = 0;
        meanAngleErrorNear = 0f; meanAngleErrorFar = 0f;
        m_T = 0f;
        m_LastSerial = bc.decisionSerial;
        isLogging = true;
        Debug.Log($"[BCExpertComparison] Start -> {m_Path}");
    }

    public void StopLogging()
    {
        if (!isLogging) return;
        isLogging = false;
        Flush();
        Debug.Log($"[BCExpertComparison] Koniec. Wierszy: {rows}\n"
                + $"  blad kata BC vs ekspert, deviation < 1 m:  {meanAngleErrorNear:F1} st  (n={nNear})\n"
                + $"  blad kata BC vs ekspert, deviation >= 1 m: {meanAngleErrorFar:F1} st  (n={nFar})\n"
                + $"  Pierwsza liczba to jakosc modelu. Druga to skala rozjechania sie "
                + "rozkladow - jesli jest znacznie wyzsza, model wpada w stany, "
                + "ktorych w danych treningowych nie bylo.");
    }

    void OnDisable() { if (isLogging) StopLogging(); }
    void OnApplicationQuit() { if (isLogging) StopLogging(); }

    void LateUpdate()
    {
        if (!isLogging || bc == null) return;
        if (bc.decisionSerial == m_LastSerial) return;
        m_LastSerial = bc.decisionSerial;

        float bcAngle = bc.lastProposedAngleDeg;
        float bcMag = bc.lastProposedMagM;
        float bcRaw = bc.lastRawArgmaxAngleDeg;

        Vector2 exp = autoExplorer.expertLocalWaypoint;
        float expAngle = autoExplorer.expertAngleDeg;
        float expMag = exp.magnitude;
        bool expOk = autoExplorer.ExpertLabelUsable;

        float angErr = Mathf.Abs(Mathf.DeltaAngle(bcAngle, expAngle));

        float r = bcAngle * Mathf.Deg2Rad;
        Vector2 bcVec = new Vector2(bcMag * Mathf.Sin(r), bcMag * Mathf.Cos(r));
        float endErr = (bcVec - exp).magnitude;

        float dev = autoExplorer.deviationFromRoute;
        float speed = (chassis != null && chassis.carRigidbody != null)
                    ? chassis.carRigidbody.linearVelocity.magnitude : 0f;


        if (expOk)
        {
            if (dev < 1f) { nNear++; m_SumNear += angErr; meanAngleErrorNear = (float)(m_SumNear / nNear); }
            else { nFar++; m_SumFar += angErr; meanAngleErrorFar = (float)(m_SumFar / nFar); }
        }

        var ci = CultureInfo.InvariantCulture;

        m_Csv.Append(bc.decisionSerial.ToString(ci)).Append(',')
             .Append(m_T.ToString("F2", ci)).Append(',')
             .Append(carTransform.position.x.ToString("F3", ci)).Append(',')
             .Append(carTransform.position.z.ToString("F3", ci)).Append(',')
             .Append(carTransform.eulerAngles.y.ToString("F2", ci)).Append(',')
             .Append(bcRaw.ToString("F2", ci)).Append(',')
             .Append(bcAngle.ToString("F2", ci)).Append(',')
             .Append(bcMag.ToString("F3", ci)).Append(',')
             .Append(bc.lastAppliedAngleDeg.ToString("F2", ci)).Append(',')
             .Append(bc.lastAppliedMagM.ToString("F3", ci)).Append(',')
             .Append(expAngle.ToString("F2", ci)).Append(',')
             .Append(expMag.ToString("F3", ci)).Append(',')
             .Append(expOk ? "1" : "0").Append(',')
             .Append(angErr.ToString("F2", ci)).Append(',')
             .Append(endErr.ToString("F3", ci)).Append(',')
             .Append(bc.lastConeMass.ToString("F4", ci)).Append(',')
             .Append(autoExplorer.progressAlongRoute.ToString("F2", ci)).Append(',')
             .Append(dev.ToString("F3", ci)).Append(',')
             .Append(speed.ToString("F3", ci)).Append(',')
             .Append(bc.lastMovedSinceDecision.ToString("F3", ci)).Append(',')
             .Append(bc.lastDecisionHeld ? "1" : "0").Append(',')
             .Append(bc.lastDecisionApplied ? "1" : "0").Append(',')
             .Append(bc.lastDecisionRecoveryStarted ? "1" : "0")
             .AppendLine();

        rows++;
        if (flushEveryRows > 0 && rows % flushEveryRows == 0) Flush();
    }

    private void Flush()
    {
        if (m_Path != null && m_Csv.Length > 0)
        {
            File.AppendAllText(m_Path, m_Csv.ToString());
            m_Csv.Clear();
        }
    }

    void OnDrawGizmos()
    {
        if (!isLogging || carTransform == null || autoExplorer == null) return;

        var yaw = Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f);
        Vector2 e = autoExplorer.expertLocalWaypoint;
        Vector3 w = carTransform.position + yaw * new Vector3(e.x, 0f, e.y);
        Gizmos.color = new Color(0f, 1f, 0.3f);
        Gizmos.DrawLine(carTransform.position, w);
        Gizmos.DrawWireCube(w, Vector3.one * 0.2f);
    }
}