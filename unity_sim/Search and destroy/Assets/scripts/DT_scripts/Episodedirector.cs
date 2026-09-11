using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Automatyczne zbieranie danych BC: losuje warunki eksperymentu dla kazdego
/// epizodu i sam zarzadza jego cyklem zycia.
///
/// Po co, skoro DTDataLogger ma auto-chunking: chunking cieie JEDEN ciagly
/// przejazd na kilka plikow. W czterech epizodach z poprzedniej sesji
/// progress_along_route biegl przez wszystkie pliki bez przerwy (54.9 -> 161.4
/// -> 161.6 -> 267.6 -> ...), czyli ostatni krok pliku N i pierwszy krok pliku
/// N+1 byly dwoma kolejnymi krokami fizyki. train_bc.py dzieli train/val po
/// grupach, wiec fragmenty TEGO SAMEGO przejazdu ladowaly po obu stronach
/// podzialu i strata walidacyjna nie mierzyla generalizacji, tylko interpolacje.
///
/// Ten komponent zmienia WARUNEK miedzy epizodami, nie tylko pozycje:
///   * punkt startu na trasie i yaw  (robi to AutoExplorer.RespawnOnRoute)
///   * kierunek jazdy                (CoverageRoute.ReverseDirection)
///   * szum eksperta wl./wyl.        (NoisyExpertDriver.active + driveTarget)
///   * dlugosc epizodu               (losowa, zeby fazy trasy sie nie powtarzaly)
///
/// UWAGA: teleport zmienia PUNKT WEJSCIA. Po kilku sekundach auto wraca na te
/// sama linie, ktora jezdzi zawsze. Rzeczywista roznorodnosc TRAJEKTORII daje
/// NoisyExpertDriver - dlatego noiseProbability jest domyslnie wysokie.
///
/// Klawisze: F1 start automatu, F2 stop, F3 wymus koniec biezacego epizodu.
/// </summary>
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

    // ------------------------------------------------------------------

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

        // trasa wraca do orientacji wyjsciowej, zeby scena nie zostala zmieniona
        if (m_RouteIsReversed) { route.ReverseDirection(); m_RouteIsReversed = false; }
    }

    private void BeginEpisode()
    {
        currentReversed = Random.value < reverseProbability;
        currentNoisy = noisyDriver != null && Random.value < noiseProbability;
        episodeTarget = Random.Range(minEpisodeSeconds, maxEpisodeSeconds);
        episodeElapsed = 0f;

        // 1. kierunek trasy - musi byc PRZED StartExploring, bo tam liczy sie
        //    routeLength i losuje punkt respawnu
        if (currentReversed != m_RouteIsReversed)
        {
            route.ReverseDirection();
            m_RouteIsReversed = currentReversed;
        }

        // 2. szum. driveTarget i noisyDriver.active musza byc rozlaczne, inaczej
        //    oba pisza po target.position i wynik zalezy od kolejnosci Update
        autoExplorer.driveTarget = !currentNoisy;
        if (noisyDriver != null) noisyDriver.active = currentNoisy;

        // 3. respawn robi AutoExplorer.StartExploring (wolane przez logger),
        //    o ile respawnOnStart = true
        autoExplorer.respawnOnStart = true;
        autoExplorer.spawnYawJitter = spawnYawJitter;

        // 4. nazwa pliku koduje warunek; czlon przed "_episode_" to grupa
        logger.instancePrefix = $"{sessionTag}_{(currentReversed ? "rev" : "fwd")}"
                              + $"_{(currentNoisy ? "noisy" : "clean")}_r{episodeIndex:D3}";

        // StartEpisode() sam wola autoExplorer.StartExploring() i czysci TofScanBuffer
        logger.StartEpisode();

        Debug.Log($"[EpisodeDirector] Epizod {episodeIndex}: "
                + $"{(currentReversed ? "ODWROTNIE" : "normalnie")}, "
                + $"{(currentNoisy ? "Z SZUMEM" : "czysty")}, "
                + $"{episodeTarget:F0} s, prefiks {logger.instancePrefix}");
    }

    private void FinishEpisode(bool keep)
    {
        bool tooShort = episodeElapsed < minKeepSeconds;
        // EndEpisode() sam wola autoExplorer.StopExploring()
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