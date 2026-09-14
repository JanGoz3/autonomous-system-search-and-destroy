using UnityEngine;
using System.Net.Sockets;
using System.Text;

public class ARUDPStreamer : MonoBehaviour
{
    public string ipAddress = "192.168.0.0";
    public int port = 5005;

    private UdpClient udpClient;
    private Vector3 positionOffset = Vector3.zero;
    private Quaternion rotationOffset = Quaternion.identity;
    private string statusMsg = "Waiting...";

    private bool isScreenBlack = true;
    private Texture2D blackTex;
    private Texture2D transparentTex;

    private Vector3 currentPos;

    void Start()
    {
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
        udpClient = new UdpClient();
        Application.targetFrameRate = 30;

        blackTex = new Texture2D(1, 1);
        blackTex.SetPixel(0, 0, Color.black);
        blackTex.Apply();

        transparentTex = new Texture2D(1, 1);
        transparentTex.SetPixel(0, 0, new Color(0, 0, 0, 0));
        transparentTex.Apply();
    }

    void Update()
    {
        Vector3 rawPos = transform.position;
        Quaternion rawRot = transform.rotation;

        Vector3 calibratedPos = Quaternion.Inverse(rotationOffset) * (rawPos - positionOffset);
        Quaternion calibratedRot = Quaternion.Inverse(rotationOffset) * rawRot;

        currentPos = calibratedPos;

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
        int padding = 40;

        GUI.DrawTexture(new Rect(0, 0, w, h), isScreenBlack ? blackTex : transparentTex);

        GUI.skin.label.fontSize = h / 40;
        GUI.skin.textField.fontSize = h / 40;
        GUI.skin.button.fontSize = h / 30;
        GUI.skin.box.fontSize = h / 35; 
        GUI.contentColor = Color.white;

        Rect coordsRect = new Rect(padding, padding, w - 2 * padding, btnHeight);
        
        int topOffset = padding + btnHeight + 20;
        Rect ipLabelRect = new Rect(padding, topOffset, w, btnHeight);
        Rect ipFieldRect = new Rect(padding, topOffset + (h / 25), w - 2 * padding, btnHeight);
        Rect statusRect = new Rect(padding, topOffset + btnHeight + (h / 25), w, btnHeight);
        
        Rect calibrateBtnRect = new Rect(padding, (h / 2) - (btnHeight / 2), w - 2 * padding, btnHeight * 2);

        Event e = Event.current;
        if (e.type == EventType.MouseDown)
        {
            Vector2 clickPos = e.mousePosition;

            if (!ipFieldRect.Contains(clickPos) && !calibrateBtnRect.Contains(clickPos))
            {
                isScreenBlack = !isScreenBlack;
            }
        }

        string coordsMsg = string.Format("X: {0:F2}  |  Y: {1:F2}  |  Z: {2:F2}", currentPos.x, currentPos.y, currentPos.z);
        GUI.Box(coordsRect, "\n" + coordsMsg);

        GUI.Label(ipLabelRect, "Jetson IP Address:");
        ipAddress = GUI.TextField(ipFieldRect, ipAddress);
        GUI.Label(statusRect, statusMsg);

        if (GUI.Button(calibrateBtnRect, "SET ZERO\n(CALIBRATE)"))
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