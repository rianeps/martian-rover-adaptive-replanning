using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Talks to TCP server. Each time the rover reaches a waypoint,
/// this captures a real frame from the rover's onboard camera (Up_Look_Camera),
/// sends it to Python as "NEXT:<base64 jpeg>\n", and receives back a JSON line
/// describing the next waypoint, terrain severity, and whether a replan happened.
///
/// This does NOT replace RoverController's ground-snapping/slope-alignment logic —
/// it only decides WHERE the rover should move to next. Movement itself should
/// still go through the same footprint-sampling raycast approach, so wire this
/// component's waypoints into RoverController rather than moving the transform here directly.
/// </summary>
public class RoverNetworkClient : MonoBehaviour
{
    [Header("Server connection")]
    public string host = "127.0.0.1";
    public int port = 9000;

    [Header("Onboard camera")]
    [Tooltip("The Camera component to capture frames from (attach a real Camera to Up_Look_Camera first).")]
    public Camera onboardCamera;
    public int captureWidth = 1280;
    public int captureHeight = 960;
    [Range(1, 100)] public int jpegQuality = 75;

    [Header("Degradation (optional — mirrors SET_DEG on the server)")]
    public string degradationType = "gaussian"; // gaussian | occlusion | dropout
    [Range(0f, 1f)] public float degradationSeverity = 0.5f;

    private TcpClient client;
    private NetworkStream stream;
    private StringBuilder recvBuffer = new StringBuilder();
    private readonly object lockObj = new object();

    // Populated from the server's response — read these from RoverController or a UI script.
    public struct WaypointResult
    {
        public bool valid;
        public string type;       // "waypoint" | "complete" | "error" | "ack" | "status"
        public float unityX, unityY, unityZ;
        public string terrain;
        public string severity;   // Clean | Degraded | Critical
        public float ssim;
        public float confidence;
        public bool isReplan;
        public int replans;
        public string message;
    }

    void Start()
    {
        Connect();
        if (client != null && client.Connected)
        {
            SendLine($"SET_DEG:{degradationType}:{degradationSeverity}");
            ReadLine(); // consume the ack
        }
    }

    void Connect()
    {
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();
            Debug.Log($"RoverNetworkClient: connected to {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"RoverNetworkClient: failed to connect to {host}:{port} — {e.Message}. " +
                            $"Make sure the local Python server (Notebook 08 / local_server.py) is running.");
        }
    }

    /// <summary>
    /// Captures the current onboard camera view, sends it to the server as the next
    /// step's frame, and returns the parsed response. Call this instead of a bare "NEXT"
    /// whenever the rover has reached its current waypoint and needs the next one.
    /// </summary>
    public WaypointResult RequestNextWaypoint()
    {
        if (client == null || !client.Connected)
        {
            Debug.LogWarning("RoverNetworkClient: not connected — attempting reconnect.");
            Connect();
            if (client == null || !client.Connected)
            {
                return new WaypointResult { valid = false, type = "error", message = "Not connected" };
            }
        }

        string base64Frame = CaptureFrameAsBase64();
        SendLine($"NEXT:{base64Frame}");
        string response = ReadLine();
        return ParseResponse(response);
    }

    public void ResetNavigation()
    {
        SendLine("RESET");
        ReadLine();
    }

    string CaptureFrameAsBase64()
    {
        if (onboardCamera == null)
        {
            Debug.LogWarning("RoverNetworkClient: no onboard camera assigned — sending a blank frame.");
            Texture2D blank = new Texture2D(captureWidth, captureHeight);
            byte[] blankBytes = blank.EncodeToJPG(jpegQuality);
            Destroy(blank);
            return Convert.ToBase64String(blankBytes);
        }

        // If the camera already has a persistent RenderTexture assigned (e.g. NavCamFeed,
        // set up in the Editor so a UI RawImage can show a live preview), read from that
        // directly — it's already being rendered into every frame by Unity automatically.
        // Otherwise, fall back to a temporary one-off render (works with no Editor setup,
        // but gives no live preview).
        bool usingPersistentTexture = onboardCamera.targetTexture != null;
        RenderTexture rt = usingPersistentTexture
            ? onboardCamera.targetTexture
            : RenderTexture.GetTemporary(captureWidth, captureHeight, 24);

        RenderTexture prevActive = RenderTexture.active;

        if (!usingPersistentTexture)
        {
            onboardCamera.targetTexture = rt;
            onboardCamera.Render();
        }
        RenderTexture.active = rt;

        Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();

        RenderTexture.active = prevActive;
        if (!usingPersistentTexture)
        {
            onboardCamera.targetTexture = null;
            RenderTexture.ReleaseTemporary(rt);
        }

        byte[] jpegBytes = tex.EncodeToJPG(jpegQuality);
        Destroy(tex);

        return Convert.ToBase64String(jpegBytes);
    }

    void SendLine(string line)
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(line + "\n");
            stream.Write(data, 0, data.Length);
        }
        catch (Exception e)
        {
            Debug.LogError($"RoverNetworkClient: send failed — {e.Message}");
        }
    }

    string ReadLine()
    {
        try
        {
            // Frames can be large (base64 JPEG on the way out; JSON with floats on the way back
            // is small), so a simple blocking read loop until we see '\n' is sufficient here.
            recvBuffer.Clear();
            byte[] buf = new byte[8192];
            while (true)
            {
                int read = stream.Read(buf, 0, buf.Length);
                if (read <= 0) break;
                string chunk = Encoding.UTF8.GetString(buf, 0, read);
                recvBuffer.Append(chunk);
                if (chunk.Contains("\n")) break;
            }
            return recvBuffer.ToString().Trim();
        }
        catch (Exception e)
        {
            Debug.LogError($"RoverNetworkClient: receive failed — {e.Message}");
            return null;
        }
    }

    WaypointResult ParseResponse(string json)
    {
        var result = new WaypointResult { valid = false };
        if (string.IsNullOrEmpty(json)) return result;

        try
        {
            var parsed = JsonUtility.FromJson<ServerMessage>(json);
            result.valid = true;
            result.type = parsed.type;
            result.unityX = parsed.unity_x;
            result.unityY = parsed.unity_y;
            result.unityZ = parsed.unity_z;
            result.terrain = parsed.terrain;
            result.severity = parsed.severity;
            result.ssim = parsed.ssim;
            result.confidence = parsed.confidence;
            result.isReplan = parsed.is_replan;
            result.replans = parsed.replans;
            result.message = parsed.message;
        }
        catch (Exception e)
        {
            Debug.LogError($"RoverNetworkClient: failed to parse server response: {json} — {e.Message}");
        }

        return result;
    }

    [Serializable]
    private class ServerMessage
    {
        public string type;
        public int step;
        public float unity_x;
        public float unity_y;
        public float unity_z;
        public int grid_row;
        public int grid_col;
        public string terrain;
        public string severity;
        public float ssim;
        public float confidence;
        public bool is_replan;
        public int replans;
        public string message;
    }

    void OnDestroy()
    {
        stream?.Close();
        client?.Close();
    }
}