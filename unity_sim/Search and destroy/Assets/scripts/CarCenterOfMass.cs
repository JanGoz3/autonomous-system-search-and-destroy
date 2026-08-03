using UnityEngine;

public class CarCenterOfMass : MonoBehaviour
{
    public Transform centerOfMassPoint;
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (centerOfMassPoint != null)
        {
            rb.centerOfMass = centerOfMassPoint.localPosition;
        }
    }

    void OnDrawGizmos()
    {
        if (centerOfMassPoint != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(centerOfMassPoint.position, 0.02f);
        }
    }
}