using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.AI;

public class CarAgent : Agent
{
    [Header("Hardware Link")]
    public Chassis chassis;

    [Header("Target Object")]
    public Transform Target;

    [Header("Navmesh Target Spawner")]
    public LocalNavMeshSpawner spawner;
    [Header("Collision & Recovery")]
    public float maxStuckDuration = 2.5f;
    private float m_StuckTimer = 0f;
    private bool m_IsColliding = false;

    [Header("Heuristic Smoothing (Keyboard Only)")]
    public float steeringSensitivity = 3f;
    public float speedSensitivity = 0.25f;
    public float turretSensitivity = 4f;


    [HideInInspector]
    public bool hadCollisionThisStep = false;

    private float m_currentSteering = 0f;
    private float m_currentSpeed = 0f;
    private float m_currentPitch = 0f;
    private float m_currentYaw = 0f;
    private float previousDistance = 0f;
    private float curriculumProgress = 0f;
    private float spawnRadius = 3f;
    private float maxSpawnAngle = 45;

    [Header("Training mode")]
    public bool trainingMode = true;
    public float startingStepOffset = 0f;
    public override void OnEpisodeBegin()
    {
        m_StuckTimer = 0f;
        m_IsColliding = false;
        
        if (trainingMode) {
            //curriculumProgress = Mathf.Clamp01(Academy.Instance.TotalStepCount + startingStepOffset/ 1e6f);
            curriculumProgress = 1.0f;
            if (chassis != null)
            {
                chassis.SetNeutral();
                
                if (chassis.carRigidbody != null)
                {
                    chassis.carRigidbody.linearVelocity = Vector3.zero;
                    chassis.carRigidbody.angularVelocity = Vector3.zero;
                }
            }

            Vector3 safeSpawnLocation = spawner.GetRandomSafePoint();

            transform.SetPositionAndRotation(
                safeSpawnLocation + new Vector3(0, 0.1f, 0), 
                Quaternion.Euler(0, Random.Range(0f, 360f), 0)
            );
        
            // TARGET SPAWN ##############
            bool foundValidSpawn = false;
            for (int i = 0; i < 10; i++)
            {
                float randomAngle = Random.Range(-maxSpawnAngle, maxSpawnAngle);
                Vector3 spawnDirection = Quaternion.Euler(0, randomAngle, 0) * transform.forward;
                Vector3 nearCarPosition = transform.position + (spawnDirection * Random.Range(1.0f, spawnRadius));
                
                NavMeshHit hit;
                if (NavMesh.SamplePosition(nearCarPosition, out hit, 5.0f, NavMesh.AllAreas))
                {
                    Target.position = hit.position + new Vector3(0, 0.05f, 0);
                    foundValidSpawn = true;
                    break;
                }
            }

            if (!foundValidSpawn)
            {
                Vector3 fallbackPos = spawner.GetRandomSafePoint();
                Target.position = fallbackPos + new Vector3(0, 0.05f, 0);
            }   
        }
        // ###########################

        previousDistance = Vector3.Distance(transform.position, Target.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (chassis == null) return;
        float[] telemetryData = chassis.GetTelemetryState();

        sensor.AddObservation(telemetryData);

        Vector3 relativeTargetPos = transform.InverseTransformPoint(Target.position);

        float maxArenaSize = 20f;

        sensor.AddObservation(relativeTargetPos.x / maxArenaSize);
        sensor.AddObservation(relativeTargetPos.z / maxArenaSize);
    }

    private void OnCollisionEnter(Collision collision) 
    {
        if (trainingMode && collision.gameObject.CompareTag("object")) 
        {
            m_IsColliding = true;
            AddReward(-0.5f); // Initial bump penalty; episode does not terminate
        }       
    }

    private void OnCollisionStay(Collision collision) 
    {
        if (trainingMode && collision.gameObject.CompareTag("object"))
        {
            m_IsColliding = true;
            AddReward(-0.01f); // Minor tick penalty for lingering/pressing into wall
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (trainingMode && collision.gameObject.CompareTag("object"))
        {
            m_IsColliding = false;
            m_StuckTimer = 0f;
            AddReward(0.3f); // Positive reinforcemenet for freeing itself
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float aiMotor = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float aiSteering = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float aiCamPitch = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float aiCamYaw = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        // Rewards
        float currentDistance = Vector3.Distance(transform.position, Target.position);

        if (chassis != null)
        {
            chassis.SetSpeed(aiMotor);
            chassis.SetSteering(aiSteering);
            chassis.SetCameraServos(aiCamPitch, aiCamYaw);
        }

        float cameraJitter = Mathf.Abs(aiCamPitch) + Mathf.Abs(aiCamYaw);
        AddReward(-0.0005f * cameraJitter);

        // WALL RECOVERY AND STUCK TIMER #############################

        if(trainingMode && m_IsColliding)
        {
            float speed = (chassis != null && chassis.carRigidbody != null) ? chassis.carRigidbody.linearVelocity.magnitude : 0f;
            // increment the timer only if the car is immobile
            if(speed < 0.2f)
            {
                m_StuckTimer += Time.fixedDeltaTime;
            }
            // car is stuck for too long
            if (m_StuckTimer >= maxStuckDuration)
            {
                SetReward(-3.5f);
                EndEpisode();
                return;
            }
        }
        else
        {
            m_StuckTimer = 0f;
        }
        // ############################


        // Reached Target (Big Reward)
        // Mathf.Lerp(A, B, t): Stands for "Linear Interpolation". It blends between value A and value B based on a percentage t.
        if (trainingMode && currentDistance < Mathf.Lerp(0.7f, 0.3f, curriculumProgress)) 
        {
            Vector3 directionToTarget = (Target.position - transform.position).normalized;
            float alignment = Vector3.Dot(transform.forward, directionToTarget);
            float formBonus = Mathf.Clamp01(alignment);
            float finalWinReward = 15.0f + (10.0f * formBonus);
            SetReward(finalWinReward);
            EndEpisode();
        } 
        // 3. Still playing
        else 
        {
            float distanceMoved = previousDistance - currentDistance;
            // suppress distance penalty while colliding so reversing away isn't punished.
            if (!m_IsColliding)
            {
                AddReward(distanceMoved);
            }
            if (MaxStep != 0)
            {
                AddReward(-1.0f / MaxStep);
            }
            previousDistance = currentDistance;
        }

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        var keyboard = Keyboard.current;
        
        if (keyboard == null) return;

        float targetSpeed = 0f;
        float targetSteering = 0f;
        float targetPitch = 0f;
        float targetYaw = 0f;

        if (keyboard.upArrowKey.isPressed) targetSpeed = 1f;
        if (keyboard.downArrowKey.isPressed) targetSpeed = -1f;

        if (keyboard.rightArrowKey.isPressed) targetSteering = 1f;
        if (keyboard.leftArrowKey.isPressed) targetSteering = -1f;

        if (keyboard.wKey.isPressed) targetPitch = 1f;
        if (keyboard.sKey.isPressed) targetPitch = -1f;

        if (keyboard.dKey.isPressed) targetYaw = 1f;
        if (keyboard.aKey.isPressed) targetYaw = -1f;

        m_currentSpeed = Mathf.MoveTowards(m_currentSpeed, targetSpeed, speedSensitivity * Time.deltaTime);
        m_currentSteering = Mathf.MoveTowards(m_currentSteering, targetSteering, steeringSensitivity * Time.deltaTime);
        m_currentPitch = Mathf.MoveTowards(m_currentPitch, targetPitch, turretSensitivity * Time.deltaTime);
        m_currentYaw = Mathf.MoveTowards(m_currentYaw, targetYaw, turretSensitivity * Time.deltaTime);

        continuousActionsOut[0] = m_currentSpeed;
        continuousActionsOut[1] = m_currentSteering;
        continuousActionsOut[2] = m_currentPitch;
        continuousActionsOut[3] = m_currentYaw;
    }
}