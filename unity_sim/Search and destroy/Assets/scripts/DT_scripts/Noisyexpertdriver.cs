using UnityEngine;
using UnityEngine.AI;


public class NoisyExpertDriver : MonoBehaviour
{
    [Header("References")]
    public AutoExplorer autoExplorer;
    public Transform carTransform;
    public Transform target;

    [Header("Wlacznik")]
    [Tooltip("Gdy true, ten komponent przejmuje sterowanie celem. Pamietaj zeby "
           + "ustawic AutoExplorer.driveTarget = false, inaczej oba beda pisac "
           + "po target.position i wynik bedzie zalezal od kolejnosci Update.")]
    public bool active = false;

    [Header("Szum kierunku")]
    [Tooltip("Maksymalne odchylenie kierunku od czystego waypointa, w stopniach (+/-). "
           + "25 st daje jazde wyraznie gorsza od eksperta, ale wciaz celowa - auto "
           + "zjezdza z trasy i wraca, zamiast bladzic losowo.")]
    public float maxAngleDeg = 25f;
    [Tooltip("Co ile sekund losowac nowa wartosc szumu. Za krotko = bialy szum, ktory "
           + "usrednia sie do zera i auto jedzie prawie jak ekspert. 2 s to okolo dwoch "
           + "decyzji modelu przy interwale 1.0 s.")]
    public float noiseHoldSeconds = 2f;
    [Tooltip("W ilu sekundach przechodzic do nowej wartosci. Zapobiega szarpaniu celem.")]
    public float blendSeconds = 0.5f;

    [Header("Szum dlugosci")]
    public float minScale = 0.6f;
    public float maxScale = 1.4f;

    [Header("Duze odchylenia (opcjonalne)")]
    [Tooltip("Szansa, ze zamiast zwyklego szumu wylosuje sie duze odchylenie. "
           + "Takie zdarzenia wypychaja auto do sasiednich pomieszczen - czyli tam, "
           + "gdzie etykieta wzdluz sciezki NavMesh faktycznie cos wnosi.")]
    [Range(0f, 1f)] public float bigDeviationChance = 0.15f;
    public float bigDeviationDeg = 70f;

    [Header("NavMesh")]
    [Tooltip("Rzutuje zaszumiony cel na NavMesh. Bez tego cel czesto laduje w scianie "
           + "i auto stoi zamiast jechac - a chcemy zeby jezdzilo ZLE, nie zeby stalo.")]
    public bool projectOntoNavMesh = true;
    public float navSampleRadius = 1.5f;

    [Header("Runtime (read-only)")]
    public float currentAngleDeg = 0f;
    public float currentScale = 1f;
    [Tooltip("Udzial klatek, w ktorych zaszumiony cel wypadl poza NavMesh.")]
    public float offMeshRate = 0f;
    [Tooltip("Udzial klatek, w ktorych sciezka NavMesh eksperta miala zalamanie, "
           + "czyli czysty ekspert prowadzil auto WOKOL przeszkody. To jest miara "
           + "tego, ile nowa etykieta faktycznie wnosi wzgledem linii prostej.")]
    public float detourRate = 0f;

    private float holdTimer = 0f;
    private float angleFrom, angleTo, scaleFrom, scaleTo;
    private float blendT = 1f;
    private int frames = 0, offMesh = 0, detours = 0;

    void Reset()
    {
        autoExplorer = GetComponent<AutoExplorer>();
        carTransform = transform;
        if (autoExplorer != null) target = autoExplorer.target;
    }

    void OnEnable()
    {
        frames = 0; offMesh = 0; detours = 0;
        holdTimer = 0f; blendT = 1f;
        angleFrom = angleTo = 0f;
        scaleFrom = scaleTo = 1f;
    }

    void Update()
    {
        if (!active || autoExplorer == null || !autoExplorer.isExploring) return;
        if (carTransform == null || target == null) return;

        if (autoExplorer.driveTarget)
        {
            Debug.LogWarning("[NoisyExpertDriver] AutoExplorer.driveTarget = true. "
                + "Oba komponenty pisza po target.position. Odznacz driveTarget.", this);
            return;
        }

        holdTimer -= Time.deltaTime;
        if (holdTimer <= 0f)
        {
            holdTimer = Mathf.Max(0.1f, noiseHoldSeconds);
            angleFrom = currentAngleDeg;
            scaleFrom = currentScale;

            float span = (Random.value < bigDeviationChance) ? bigDeviationDeg : maxAngleDeg;
            angleTo = Random.Range(-span, span);
            scaleTo = Random.Range(minScale, maxScale);
            blendT = 0f;
        }

        if (blendSeconds > 0.01f) blendT = Mathf.Min(1f, blendT + Time.deltaTime / blendSeconds);
        else blendT = 1f;

        currentAngleDeg = Mathf.Lerp(angleFrom, angleTo, blendT);
        currentScale = Mathf.Lerp(scaleFrom, scaleTo, blendT);

        Vector2 clean = autoExplorer.expertLocalWaypoint;
        float th = currentAngleDeg * Mathf.Deg2Rad;
        float c = Mathf.Cos(th), s = Mathf.Sin(th);

        Vector2 noisy = new Vector2(clean.x * c + clean.y * s,
                                    clean.y * c - clean.x * s) * currentScale;

        Vector3 world = carTransform.position
                      + Quaternion.Euler(0f, carTransform.eulerAngles.y, 0f)
                        * new Vector3(noisy.x, 0f, noisy.y);

        frames++;
        if (projectOntoNavMesh)
        {
            if (NavMesh.SamplePosition(world, out NavMeshHit hit, navSampleRadius, NavMesh.AllAreas))
                world = hit.position;
            else
                offMesh++;
        }
        if (!autoExplorer.expertPathValid) { /* etykieta i tak bedzie odfiltrowana */ }

        float straight = autoExplorer.expertLocalWaypoint.magnitude;
        if (straight > 0.05f && autoExplorer.pathLengthToGoal > straight * 1.3f) detours++;

        offMeshRate = frames > 0 ? (float)offMesh / frames : 0f;
        detourRate = frames > 0 ? (float)detours / frames : 0f;

        target.position = world + new Vector3(0f, 0.15f, 0f);
    }

    void OnDrawGizmos()
    {
        if (!active || !Application.isPlaying || carTransform == null || target == null) return;
        Gizmos.color = new Color(1f, 0.5f, 0f);   // pomaranczowy = cel ZASZUMIONY
        Gizmos.DrawLine(carTransform.position, target.position);
        Gizmos.DrawWireSphere(target.position, 0.25f);
    }
}