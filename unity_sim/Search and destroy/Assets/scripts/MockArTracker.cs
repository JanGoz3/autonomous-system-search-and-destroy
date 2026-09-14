using UnityEngine;
using System.Collections.Generic;

public class MockArTracker : MonoBehaviour
{
    [Header("Noise & Drift Profile")]
    public float positionJitterStdDev = 0.008f;
    public float yawDriftDegPer100m = 2.0f;
    public float scaleErrorFactor = 0.02f;
    [Header("Latency Simulation")]
    public float latencySeconds = 0.04f;

    private Vector3 m_SessionOriginPosition;
    private Quaternion m_SessionOriginRotation;
    private Vector3 m_LastTruePosition;
    private float m_AccumulatedYawDrift = 0f;
    private float m_RandomScaleDrift = 1.0f;
    private float m_SessionDriftDirection = 1.0f;

    private struct PoseRecord
    {
        public float timestamp;
        public Vector3 position;
        public float yaw;
    }
    private readonly Queue<PoseRecord> m_PoseHistory = new Queue<PoseRecord>();

    public void ResetSession()
    {
        m_SessionOriginPosition = transform.position;
        m_SessionOriginRotation = transform.rotation;
        m_LastTruePosition = transform.position;

        m_AccumulatedYawDrift = 0f;

        // Choose consistent drift direction and scale imperfection for this run
        m_SessionDriftDirection = Random.value > 0.5f ? 1f : -1f;

        m_RandomScaleDrift = 1.0f + Random.Range(-scaleErrorFactor, scaleErrorFactor);
        m_PoseHistory.Clear();
    }

    void FixedUpdate()
    {
        Vector3 trueDeltaWorld = transform.position - m_SessionOriginPosition;
        Vector3 sessionPos = Quaternion.Inverse(m_SessionOriginRotation) * trueDeltaWorld;

        float trueYaw = (Quaternion.Inverse(m_SessionOriginRotation) * transform.rotation).eulerAngles.y;
        if (trueYaw > 180f) trueYaw -= 360f;

        float stepDist = Vector3.Distance(transform.position, m_LastTruePosition);
        m_LastTruePosition = transform.position;

        m_AccumulatedYawDrift += stepDist / 100f * yawDriftDegPer100m * m_SessionDriftDirection;

        // Inject imperfections
        // Scale error + high frequency Gaussian-like jitter
        float jitterX = (Random.Range(-1f, 1f) + Random.Range(-1f, 1f)) * 0.5f * positionJitterStdDev;
        float jitterZ = (Random.Range(-1f, 1f) + Random.Range(-1f, 1f)) * 0.5f * positionJitterStdDev;

        Vector3 noisySessionPos = new Vector3(
            (sessionPos.x * m_RandomScaleDrift) + jitterX,
            0f,
            (sessionPos.z * m_RandomScaleDrift) + jitterZ
        );

        float noisyYaw = trueYaw + m_AccumulatedYawDrift + Random.Range(-0.2f, 0.2f);

        m_PoseHistory.Enqueue(new PoseRecord
        {
            timestamp = Time.time,
            position = noisySessionPos,
            yaw = noisyYaw
        });

        while (m_PoseHistory.Count > 50)
        {
            m_PoseHistory.Dequeue();
        }
    }

    /// <summary>
    /// Calculates the remaining distance to the waypoint from the car's local perspective 
    /// (X = right/left, Z = forward/backward), exactly how a physical phone's AR tracker 
    /// would calculate it in the real world.
    /// </summary>
    public Vector2 GetRelativeTargetVector(Vector3 targetWorldPosition)
    {
        float targetTime = Time.time - latencySeconds;
        PoseRecord currentPose = m_PoseHistory.Count > 0 ? m_PoseHistory.Peek() : default;

        while (m_PoseHistory.Count > 1 && m_PoseHistory.Peek().timestamp < targetTime)
        {
            currentPose = m_PoseHistory.Dequeue();
        } 

        // Convert target to Session Space
        Vector3 targetDeltaWorld = targetWorldPosition - m_SessionOriginPosition;
        Vector3 targetSessionPos = Quaternion.Inverse(m_SessionOriginRotation) * targetDeltaWorld;

        // rotate by the car's noisy heading (-yaw) to bring into car local space
        Vector3 sessionDelta = targetSessionPos - currentPose.position;
        Vector3 carLocal = Quaternion.Euler(0f, -currentPose.yaw, 0f) * sessionDelta;
        return new Vector2(carLocal.x, carLocal.z);
    }
}
