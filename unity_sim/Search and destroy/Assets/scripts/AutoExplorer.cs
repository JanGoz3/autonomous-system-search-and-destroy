using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class AutoExplorer : MonoBehaviour
{
    [Header("References")]
    public Transform target;
    public Transform carTransform;
    public LocalNavMeshSpawner spawner;

    [Header("Exploration Settings")]
    [Tooltip("Minimalna/maksymalna odleglosc losowanego celu (punktu B) od miejsca, z ktorego zaczynamy planowac trase")]
    public float minTargetDistance = 2f;
    public float maxTargetDistance = 3f;
    [Tooltip("Jak szybko (m/s) target przesuwa sie wzdluz wyznaczonej sciezki. Dostosuj do predkosci drivera - zbyt szybki target = driver ciagle 'goni', zbyt wolny = driver czeka.")]
    public float targetMoveSpeed = 2f;
    [Tooltip("Jak blisko naroznika sciezki target musi byc, zeby uznac ze go osiagnal i przejsc do kolejnego")]
    public float arrivalThreshold = 0.3f;

    [Header("Stuck Detection")]
    [Tooltip("Co ile sekund sprawdzamy czy AUTO (nie target) w ogole sie rusza")]
    public float stuckCheckWindow = 3f;
    [Tooltip("Jesli auto przemiescilo sie mniej niz to (w metrach) w ciagu stuckCheckWindow, uznajemy ze utknelo i przeplanowujemy sciezke od jego FAKTYCZNEJ pozycji")]
    public float stuckDistanceThreshold = 0.3f;

    [Header("Runtime State")]
    public bool isExploring = false;

    private List<Vector3> pathCorners = new List<Vector3>();
    private int currentCornerIndex = 0;

    private float stuckTimer = 0f;
    private Vector3 lastCheckPosition;

    void Update()
    {
        if (!isExploring) return;

        stuckTimer += Time.deltaTime;

        if (stuckTimer >= stuckCheckWindow)
        {
            float moved = Vector3.Distance(carTransform.position, lastCheckPosition);
            if (moved < stuckDistanceThreshold)
            {
                PlanNewPath(carTransform.position);
            }
            stuckTimer = 0f;
            lastCheckPosition = carTransform.position;
        }

        MoveTargetAlongPath();
    }

    private void MoveTargetAlongPath()
    {
        if (pathCorners.Count == 0 || currentCornerIndex >= pathCorners.Count)
        {
            PlanNewPath(target.position);
            return;
        }

        Vector3 nextCorner = pathCorners[currentCornerIndex];
        target.position = Vector3.MoveTowards(target.position, nextCorner, targetMoveSpeed * Time.deltaTime);

        if (Vector3.Distance(target.position, nextCorner) < arrivalThreshold)
        {
            currentCornerIndex++;
        }
    }

    private void PlanNewPath(Vector3 fromPosition)
    {
        Vector3 destination = PickRandomDestination(fromPosition);

        NavMeshPath navPath = new NavMeshPath();
        bool success = NavMesh.CalculatePath(fromPosition, destination, NavMesh.AllAreas, navPath);

        if (success && navPath.status == NavMeshPathStatus.PathComplete && navPath.corners.Length > 0)
        {
            pathCorners = new List<Vector3>();
            foreach (Vector3 corner in navPath.corners)
            {
                pathCorners.Add(corner + new Vector3(0, 0.05f, 0));
            }

            target.position = pathCorners[0];
            currentCornerIndex = pathCorners.Count > 1 ? 1 : 0;
        }
        else
        {

            NavMeshHit hit;
            if (NavMesh.SamplePosition(destination, out hit, 3.0f, NavMesh.AllAreas))
            {
                pathCorners = new List<Vector3> { hit.position + new Vector3(0, 0.05f, 0) };
            }
            else
            {
                pathCorners = new List<Vector3> { spawner.GetRandomSafePoint() + new Vector3(0, 0.05f, 0) };
            }
            currentCornerIndex = 0;
        }
    }

    private Vector3 PickRandomDestination(Vector3 fromPosition)
    {
        for (int i = 0; i < 10; i++)
        {
            float dist = Random.Range(minTargetDistance, maxTargetDistance);
            Vector2 randomCircle = Random.insideUnitCircle.normalized * dist;
            Vector3 candidate = fromPosition + new Vector3(randomCircle.x, 0, randomCircle.y);

            NavMeshHit hit;
            if (NavMesh.SamplePosition(candidate, out hit, 3.0f, NavMesh.AllAreas))
            {
                return hit.position;
            }
        }
        return spawner.GetRandomSafePoint();
    }

    public void StartExploring()
    {
        isExploring = true;
        stuckTimer = 0f;
        lastCheckPosition = carTransform.position;
        PlanNewPath(carTransform.position);
    }

    public void StopExploring()
    {
        isExploring = false;
    }
}