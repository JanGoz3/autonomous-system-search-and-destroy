using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class EpisodeDirector : MonoBehaviour
{
    [Header("References")]
    public DTDataLogger logger;
    public AutoExplorer autoExplorer;
    public NoisyExpertDriver noisyDriver;
    public CoverageRoute route;
    public CarAgent carAgent;

    [Header("Plan sesji")]
    [Tooltip("Ile epizodow zebrac, potem automat sam sie zatrzyma. 0 = bez limitu.")]
    public int episodesToCollect = 16;
    [Tooltip("Minimalna dlugosc epizodu w sekundach.")]
    public float minEpisodeSeconds = 120f;
    [Tooltip("Maksymalna dlugosc epizodu. Losowa dlugosc zamiast 'do konca okrazenia' - "
           + "inaczej kazdy epizod ma te sama dlugosc i punkty startu sa skorelowane "
           + "z geometria petli.")]
    public float maxEpisodeSeconds = 240f;

    [Header("Losowane warunki")]
    [Tooltip("Prawdopodobienstwo, ze epizod bedzie jechal w ODWROTNYM kierunku trasy. "
           + "0.5 = polowa sesji w kazda strone. Najtansza dzwignia roznorodnosci: "
           + "podwaja liczbe roznych stanow i psuje memoryzacje absolutnego yaw.")]
    [Range(0f, 1f)] public float reverseProbability = 0.5f;
    [Tooltip("Prawdopodobienstwo wlaczenia NoisyExpertDriver. To glowne zrodlo probek "
           + "uczacych odzyskiwania - bez nich model w inferencji przy pierwszym "
           + "odchyleniu wychodzi poza rozklad treningowy.")]
    [Range(0f, 1f)] public float noiseProbability = 0.65f;
    [Tooltip("Jitter yaw przy respawnie, przekazywany do AutoExplorer. Trzymaj ponizej "
           + "maxLabelAngleDeg, inaczej pierwsze kroki epizodu maja etykiete poza stozkiem.")]
    public float spawnYawJitter = 45f;

    [Header("Rownomierne punkty startu")]
    [Tooltip("Losowanie WARSTWOWE zamiast i.i.d.: trasa dzielona na kubelki, kazdy epizod "
           + "dostaje kolejny z przetasowanej listy, po wyczerpaniu lista tasuje sie od nowa. "
           + "AutoExplorer.RespawnOnRoute losuje Random.Range(0, routeLength), co przy "
           + "kilkudziesieciu epizodach daje zbitki i luki - w pierwszych osmiu epizodach "
           + "piec startow wypadlo w pierwszej trzeciej 56-metrowej trasy.")]
    public bool uniformStartPoints = true;
    [Tooltip("Na ile kubelkow podzielic trase. Przy 56 m i 12 kubelkach jeden ma ~4.7 m, "
           + "czyli kazdy narozník trafia do innego kubelka.")]
    [Min(2)] public int startBins = 12;

    [Header("Nazewnictwo plikow")]
    [Tooltip("Prefiks kodujacy warunek, np. bc_fwd_clean_r003. build_bc_dataset.py "
           + "grupuje po czlonie przed '_episode_', wiec podzial train/val idzie po SESJI.")]
    public string sessionTag = "bc";

    [Header("Odrzucanie")]
    [Tooltip("Epizod krotszy niz tyle sekund jest odrzucany zamiast zapisywany "
           + "(np. gdy trzeba bylo przerwac recznie).")]
    public float minKeepSeconds = 30f;

    [Header("Runtime (read-only)")]
    public bool isRunning = false;
    public int episodeIndex = 0;
    public float episodeElapsed = 0f;
    public float episodeTarget = 0f;
    public bool currentReversed = false;
    public bool currentNoisy = false;
    [Tooltip("Historia zebranych epizodow - do szybkiego sprawdzenia, czy warunki "
           + "faktycznie sie mieszaly.")]
    public List<string> collected = new List<string>();

    private bool m_RouteIsReversed = false;
    private readonly List<int> m_BinQueue = new List<int>();

    void Reset()
    {
        logger = GetComponent<DTDataLogger>();
        autoExplorer = GetComponent<AutoExplorer>();
        noisyDriver = GetComponent<NoisyExpertDriver>();
        carAgent = GetComponent<CarAgent>();
        if (autoExplorer != null) route = autoExplorer.route;
    }

    void Start()
    {
        if (route == null && autoExplorer != null) route = autoExplorer.route;
    }

    void Update()
    {
        var kb = Keyboard.current;
        if (kb != null)
        {
            if (kb.f1Key.wasPressedThisFrame && !isRunning) StartSession();
            if (kb.f2Key.wasPressedThisFrame && isRunning) StopSession();
            if (kb.f3Key.wasPressedThisFrame && isRunning) FinishEpisode(keep: true);
        }
        if (!isRunning) return;

        episodeElapsed += Time.deltaTime;
        if (episodeElapsed >= episodeTarget) FinishEpisode(keep: true);
    }


    public void StartSession()
    {
        if (logger == null || autoExplorer == null || route == null)
        {
            Debug.LogError("[EpisodeDirector] Brak referencji logger/autoExplorer/route.", this);
            return;
        }
        if (carAgent != null && carAgent.trainingMode)
        {
            Debug.LogError("[EpisodeDirector] CarAgent.trainingMode = true. OnEpisodeBegin "
                + "teleportuje auto i przestawia Target, walczac z AutoExplorer. "
                + "Odznacz trainingMode.", this);
            return;
        }
        if (logger.autoEndAfterSteps != 0)
            Debug.LogWarning($"[EpisodeDirector] DTDataLogger.autoEndAfterSteps = "
                + $"{logger.autoEndAfterSteps}. Automat sam konczy epizody, wiec chunking "
                + "tylko pociebie je dodatkowo. Ustaw 0.", this);

        episodeIndex = 0;
        collected.Clear();
        m_BinQueue.Clear();
        isRunning = true;
        BeginEpisode();
    }

    public void StopSession()
    {
        if (!isRunning) return;
        FinishEpisode(keep: episodeElapsed >= minKeepSeconds);
        isRunning = false;

        int rev = 0, noisy = 0;
        foreach (string s in collected)
        {
            if (s.Contains("rev")) rev++;
            if (s.Contains("noisy")) noisy++;
        }
        Debug.Log($"[EpisodeDirector] Koniec sesji. Epizodow: {collected.Count}  "
                + $"odwroconych: {rev}  z szumem: {noisy}");

        if (m_RouteIsReversed) { route.ReverseDirection(); m_RouteIsReversed = false; }
    }

    private void BeginEpisode()
    {
        currentReversed = Random.value < reverseProbability;
        currentNoisy = noisyDriver != null && Random.value < noiseProbability;
        episodeTarget = Random.Range(minEpisodeSeconds, maxEpisodeSeconds);
        episodeElapsed = 0f;

        if (currentReversed != m_RouteIsReversed)
        {
            route.ReverseDirection();
            m_RouteIsReversed = currentReversed;
        }

        autoExplorer.driveTarget = !currentNoisy;
        if (noisyDriver != null) noisyDriver.active = currentNoisy;

        autoExplorer.respawnOnStart = true;
        autoExplorer.spawnYawJitter = spawnYawJitter;

        float startD = -1f;
        if (uniformStartPoints)
        {
            float len = route.TotalLength();
            if (len > 0.1f)
            {
                int bin = NextBin();
                startD = (bin + Random.value) * len / startBins;
            }
        }
        autoExplorer.forcedSpawnDistance = startD;

        logger.instancePrefix = $"{sessionTag}_{(currentReversed ? "rev" : "fwd")}"
                              + $"_{(currentNoisy ? "noisy" : "clean")}_r{episodeIndex:D3}";

        logger.StartEpisode();

        Debug.Log($"[EpisodeDirector] Epizod {episodeIndex}: "
                + $"{(currentReversed ? "ODWROTNIE" : "normalnie")}, "
                + $"{(currentNoisy ? "Z SZUMEM" : "czysty")}, "
                + $"{episodeTarget:F0} s, start "
                + (startD >= 0f ? $"{startD:F1} m (kubelek {startBins - m_BinQueue.Count}/{startBins})"
                                : "losowy")
                + $", prefiks {logger.instancePrefix}");
    }

    private int NextBin()
    {
        if (m_BinQueue.Count == 0)
        {
            for (int i = 0; i < startBins; i++) m_BinQueue.Add(i);
            for (int i = m_BinQueue.Count - 1; i > 0; i--)   // tasowanie Fishera-Yatesa
            {
                int j = Random.Range(0, i + 1);
                (m_BinQueue[i], m_BinQueue[j]) = (m_BinQueue[j], m_BinQueue[i]);
            }
        }
        int bin = m_BinQueue[m_BinQueue.Count - 1];
        m_BinQueue.RemoveAt(m_BinQueue.Count - 1);
        return bin;
    }

    private void FinishEpisode(bool keep)
    {
        bool tooShort = episodeElapsed < minKeepSeconds;
        logger.EndEpisode(discard: !keep || tooShort);

        if (keep && !tooShort)
        {
            collected.Add(logger.instancePrefix);
            Debug.Log($"[EpisodeDirector] Zapisano epizod {episodeIndex} "
                    + $"({episodeElapsed:F0} s). Etykiety nieuzywalne: "
                    + $"{100f * autoExplorer.labelRejectRate:F1}%  "
                    + $"zaklinowania: {autoExplorer.stuckEvents}");
            episodeIndex++;
        }
        else
        {
            Debug.Log($"[EpisodeDirector] Odrzucono epizod ({episodeElapsed:F0} s "
                    + $"< {minKeepSeconds} s)");
        }

        if (!isRunning) return;

        if (episodesToCollect > 0 && episodeIndex >= episodesToCollect)
        {
            isRunning = false;
            if (m_RouteIsReversed) { route.ReverseDirection(); m_RouteIsReversed = false; }
            Debug.Log($"[EpisodeDirector] Zebrano {episodeIndex} epizodow - koniec planu.");
            return;
        }

        BeginEpisode();
    }
}