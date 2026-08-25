using UnityEngine;
using UnityEngine.AI;

public class AutoExplorer : MonoBehaviour
{
    [Header("References")]
    public Transform target; // ten sam obiekt Target co CarAgent używa
    public Transform carTransform;
    public LocalNavMeshSpawner spawner;

    [Header("Exploration Settings")]
    public float retargetInterval = 6f; // co ile sekund losować nowy cel
    public float minTargetDistance = 5f; // minimalna odległość nowego celu od auta
    public float maxTargetDistance = 5f; // maksymalna odległość nowego celu od auta;

    [Header("Stuck Detection")]
    [Tooltip("Co ile sekund sprawdzamy czy auto sie w ogole rusza")]
    public float stuckCheckWindow = 3f;
    [Tooltip("Jesli auto przemiescilo sie mniej niz to (w metrach) w ciagu stuckCheckWindow, uznajemy ze utknelo i wymuszamy nowy cel NATYCHMIAST")]
    public float stuckDistanceThreshold = 0.3f;

    [Header("Runtime State")]
    public bool isExploring = false;

    private float timer = 0f;
    private float stuckTimer = 0f;
    private Vector3 lastCheckPosition;

    void Update()
    {
        if (!isExploring) return;

        timer += Time.deltaTime;
        stuckTimer += Time.deltaTime;

        // --- wykrywanie utkniecia: sprawdzamy czy auto w ogole sie przemieszcza ---
        if (stuckTimer >= stuckCheckWindow)
        {
            float moved = Vector3.Distance(carTransform.position, lastCheckPosition);

            if (moved < stuckDistanceThreshold)
            {
                // auto utkneło (np. cel byl za sciana) - wymuszamy nowy cel od razu,
                // nie czekajac na pelny retargetInterval
                timer = 0f;
                PickNewTarget();
            }

            stuckTimer = 0f;
            lastCheckPosition = carTransform.position;
        }

        if (timer >= retargetInterval)
        {
            timer = 0f;
            PickNewTarget();
        }
    }

    private void PickNewTarget()
    {
        for (int i = 0; i < 10; i++)
        {
            float dist = Random.Range(minTargetDistance, maxTargetDistance);
            Vector2 randomCircle = Random.insideUnitCircle.normalized * dist;
            Vector3 candidate = carTransform.position + new Vector3(randomCircle.x, 0, randomCircle.y);

            NavMeshHit hit;
            if (NavMesh.SamplePosition(candidate, out hit, 3.0f, NavMesh.AllAreas))
            {
                target.position = hit.position + new Vector3(0, 0.05f, 0);
                return;
            }
        }
        target.position = spawner.GetRandomSafePoint() + new Vector3(0, 0.05f, 0);
    }

    public void StartExploring()
    {
        isExploring = true;
        timer = retargetInterval; // wymuś natychmiastowy pierwszy cel
        stuckTimer = 0f;
        lastCheckPosition = carTransform.position;
    }

    public void StopExploring()
    {
        isExploring = false;
    }
}