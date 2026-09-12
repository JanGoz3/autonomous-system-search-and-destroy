using UnityEngine;
using System.Net.Sockets;
using System.Text;

public class ARUDPStreamer : MonoBehaviour
{
    public string ipAddress = "192.168.42.50";
    public int port = 5005;

    private UdpClient udpClient;

    private Vector3 positionOffset = Vector3.zero;
    private Quaternion rotationOffset = Quaternion.identity;

    private string statusMsg = "Waiting...";

    void Start()
    {
        udpClient = new UdpClient();
        Application.targetFrameRate = 30;
    }

    void Update()
    {
        Vector3 rawPos = transform.position;
        Quaternion rawRot = transform.rotation;

        Vector3 calibratedPos = Quaternion.Inverse(rotationOffset) * (rawPos - positionOffset);
        Quaternion calibratedRot = Quaternion.Inverse(rotationOffset) * rawRot;

        string message = string.Format(System.Globalization.CultureInfo.InvariantCulture,
            "{0:F4},{1:F4},{2:F4},{3:F4},{4:F4},{5:F4},{6:F4}",
            calibratedPos.x, calibratedPos.y, calibratedPos.z,
            calibratedRot.x, calibratedRot.y, calibratedRot.z, calibratedRot.w);

        byte[] data = Encoding.UTF8.GetBytes(message);

        try
        {
            udpClient.Send(data, data.Length, ipAddress, port);
            statusMsg = "Streaming to " + ipAddress + ":" + port;
        }
        catch (System.Exception e)
        {
            statusMsg = "ERROR: " + e.Message;
        }
    }

    void OnGUI()
    {
        int w = Screen.width;
        int h = Screen.height;
        int btnHeight = h / 8;
        int padding = 20;

        GUI.skin.label.fontSize = h / 30;
        GUI.skin.textField.fontSize = h / 30;
        GUI.skin.button.fontSize = h / 40;

        GUI.Label(new Rect(padding, padding, w, btnHeight), "Jetson IP Address:");
        ipAddress = GUI.TextField(new Rect(padding, padding + (h / 20), w - 2 * padding, btnHeight), ipAddress);

        GUI.Label(new Rect(padding, padding + 2 * btnHeight, w, btnHeight), statusMsg);

        if (GUI.Button(new Rect(padding, h - btnHeight - padding, w - 2 * padding, btnHeight), "SET ZERO (CALIBRATE)"))
        {
            CalibrateZeroPoint();
        }
    }

    private void CalibrateZeroPoint()
    {
        positionOffset = transform.position;
        rotationOffset = transform.rotation;
        
        statusMsg = "Zero point set!";
        Debug.Log("AR Calibrated Zero Point");
    }

    void OnApplicationQuit()
    {
        if (udpClient != null) udpClient.Close();
    }
}