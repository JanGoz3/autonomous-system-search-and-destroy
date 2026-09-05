using UnityEngine;

public class TofScanBuffer : MonoBehaviour
{
    [Header("References")]
    public TofSensor tofSensor;
    public ServosCamera servosCamera;

    [Header("Scan Configuration")]
    [Min(1)] public int sectorCount = 16;
    // AutonomousCar.prefab: physicalLeftYawAngle = -87.5, physicalRightYawAngle = 76.5 stopnia.
    public float minYawDegrees = -87.5f;
    public float maxYawDegrees = 76.5f;
    [Min(0.01f)] public float maxMeasurementAgeSeconds = 5f;

    [Header("Runtime (read-only)")]
    [Tooltip("Ile sektorow ma swiezy pomiar. Jesli utrzymuje sie blisko 1, wiezyczka nie omiata zakresu i profil nie powstaje.")]
    public int freshSectors = 0;

    private const float MaxDistanceMm = 4000f;
    private float[] distancesMm;
    private float[] pitchesDeg;
    private float[] measurementTimes;
    private bool[] hasMeasurement;

    public int SectorCount => Mathf.Max(1, sectorCount);

    void Reset()
    {
        servosCamera = GetComponentInParent<ServosCamera>();
        if (servosCamera == null)
            servosCamera = GetComponentInChildren<ServosCamera>(true);
        tofSensor = servosCamera != null
            ? servosCamera.GetComponentInChildren<TofSensor>(true)
            : GetComponentInChildren<TofSensor>(true);
    }

    void Awake()
    {
        Clear();
    }

    void Start()
    {
        if (tofSensor == null || servosCamera == null)
            Debug.LogWarning("[TofScanBuffer] Brak referencji TofSensor lub ServosCamera. "
                + "Podepnij oba komponenty w Inspectorze; profil pozostanie nieznany.", this);
        if (maxYawDegrees <= minYawDegrees || maxMeasurementAgeSeconds <= 0f)
            Debug.LogWarning("[TofScanBuffer] Nieprawidlowy zakres yaw lub maksymalny wiek pomiaru. "
                + "Popraw konfiguracje w Inspectorze.", this);
    }

    void FixedUpdate()
    {
        if (tofSensor == null || servosCamera == null || maxYawDegrees <= minYawDegrees
            || maxMeasurementAgeSeconds <= 0f) return;

        EnsureStorage();

        (float pitch, float yaw) angles = servosCamera.GetActualPitchYawDegrees();
        float normalizedYaw = Mathf.InverseLerp(minYawDegrees, maxYawDegrees, angles.yaw);
        int sector = Mathf.Min(Mathf.FloorToInt(normalizedYaw * SectorCount), SectorCount - 1);

        distancesMm[sector] = tofSensor.GetDistance();
        pitchesDeg[sector] = angles.pitch;
        measurementTimes[sector] = Time.time;
        hasMeasurement[sector] = true;

        RefreshFreshCount();
    }

    public void Clear()
    {
        distancesMm = new float[SectorCount];
        pitchesDeg = new float[SectorCount];
        measurementTimes = new float[SectorCount];
        hasMeasurement = new bool[SectorCount];
        freshSectors = 0;
    }

    private void EnsureStorage()
    {
        if (distancesMm == null || distancesMm.Length != SectorCount)
            Clear();
    }

    private bool IsFresh(int sector, float now) =>
        hasMeasurement[sector] && now - measurementTimes[sector] < maxMeasurementAgeSeconds;

    private void RefreshFreshCount()
    {
        float now = Time.time;
        int n = 0;
        for (int s = 0; s < SectorCount; s++) if (IsFresh(s, now)) n++;
        freshSectors = n;
    }

    public float[] GetNormalizedDistances()
    {
        EnsureStorage();
        float now = Time.time;
        float[] snapshot = new float[SectorCount];
        for (int s = 0; s < snapshot.Length; s++)
            if (IsFresh(s, now))
                snapshot[s] = Mathf.Clamp01(distancesMm[s] / MaxDistanceMm);
        return snapshot;
    }

    public float[] GetNormalizedAges()
    {
        EnsureStorage();
        float now = Time.time;
        float[] snapshot = new float[SectorCount];
        for (int s = 0; s < snapshot.Length; s++)
            snapshot[s] = hasMeasurement[s]
                ? Mathf.Clamp01((now - measurementTimes[s]) / maxMeasurementAgeSeconds)
                : 1f;
        return snapshot;
    }

    public float[] GetMeasurementPitchesDegrees()
    {
        EnsureStorage();
        float now = Time.time;
        float[] snapshot = new float[SectorCount];
        for (int s = 0; s < snapshot.Length; s++)
            if (IsFresh(s, now))
                snapshot[s] = pitchesDeg[s];
        return snapshot;
    }
}