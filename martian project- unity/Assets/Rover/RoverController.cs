using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Moves the rover along the waypoints exported from  path_data.json.
/// X and Z come from the JSON (they're already correctly scaled/offset to match
/// the terrain's real world bounds). Y is NOT taken from the JSON — it's snapped
/// at runtime via raycast against the terrain's Mesh Collider, since that's more
/// accurate than the notebook's coarse 30x30 elevation grid and avoids re-deriving
/// the terrain's rotation math on the Python side.
/// </summary>
public class RoverController : MonoBehaviour
{
    [Header("Path data")]
    [Tooltip("Path relative to StreamingAssets, e.g. path_data.json")]
    public string jsonFileName = "path_data.json";

    [Header("Movement")]
    public float moveSpeed = 500f;          // units/second — tune to your terrain scale
    public float arrivalThreshold = 5f;     // how close counts as "reached" a waypoint
    public bool loop = false;

    [Header("Ground snapping")]
    public float raycastStartHeight = 10000f; // start well above the tallest terrain point
    public float raycastMaxDistance = 20000f;
    public float groundOffset = 0f;           // raise the rover slightly above the surface if needed
    public LayerMask groundLayerMask = ~0;    // restrict this if you have other colliders in the scene

    [Header("Slope alignment")]
    [Tooltip("If false, the rover stays flat and only its height changes — it will look like it's gliding over bumps.")]
    public bool alignToSlope = true;
    public float slopeAlignSpeed = 4f;        // higher = rover snaps to slope faster
    [Tooltip("Hard cap on how fast the rover's body can physically turn, in degrees per second — real rovers turn slowly (a few deg/sec), this prevents unrealistic snap-turns at sharp waypoint direction changes.")]
    public float maxTurnDegreesPerSecond = 15f;

    private bool loggedMissedRaycast = false;
    private float pivotToBaseOffset = 0f; // computed at Start — see ComputePivotOffset()
    private float halfLength = 1f;        // half the rover's footprint along its forward axis
    private float halfWidth = 1f;         // half the rover's footprint along its right axis

    [Header("Suspension sampling")]
    [Tooltip("How much of the rover's own footprint to sample at (1 = full half-length/half-width, lower = more conservative).")]
    [Range(0.2f, 1f)]
    public float footprintSampleFraction = 0.8f;

    [Header("Live network navigation")]
    [Tooltip("If true, waypoints come from RoverNetworkClient (live server) instead of the static JSON file below.")]
    public bool useNetworking = true;

    private RoverNetworkClient networkClient;
    private Vector2? currentTargetXZ = null;
    private bool navigationComplete = false;
    private bool waitingForServer = false;

    private List<Vector2> waypointsXZ = new List<Vector2>();
    private int currentIndex = 0;

    [System.Serializable]
    private class Waypoint
    {
        public int step;
        public int grid_row;
        public int grid_col;
        public float unity_x;
        public float unity_y;
        public float unity_z;
        public string terrain;
        public float elevation;
        public bool is_replan;
        public string severity;
    }

    [System.Serializable]
    private class PathData
    {
        public List<Waypoint> waypoints;
    }

    void Start()
    {
        ComputePivotOffset();

        if (useNetworking)
        {
            networkClient = GetComponent<RoverNetworkClient>();
            if (networkClient == null)
            {
                Debug.LogError("RoverController: Use Networking is enabled but no RoverNetworkClient " +
                                "component was found on this object. Add one, or disable Use Networking " +
                                "to fall back to the static path_data.json.");
                return;
            }
            RequestNextFromServer(); // fetches the first waypoint and snaps to it instantly
            if (currentTargetXZ.HasValue)
            {
                SnapToPosition(currentTargetXZ.Value);
            }
        }
        else
        {
            LoadPath();
            if (waypointsXZ.Count > 0)
            {
                SnapToWaypoint(0, instant: true);
            }
        }
    }

    /// <summary>
    /// Asks the live server for the next waypoint (sending the rover's current onboard
    /// camera frame along with the request). Updates currentTargetXZ, or marks navigation
    /// complete/paused depending on the response type.
    /// </summary>
    void RequestNextFromServer()
    {
        waitingForServer = true;
        var result = networkClient.RequestNextWaypoint();
        waitingForServer = false;

        if (!result.valid)
        {
            Debug.LogWarning("RoverController: server request failed or returned nothing — will retry next arrival.");
            return;
        }

        switch (result.type)
        {
            case "waypoint":
                currentTargetXZ = new Vector2(result.unityX, result.unityZ);
                string replanTag = result.isReplan ? " [REPLANNED]" : "";
                Debug.Log($"RoverController: step target ({result.unityX}, {result.unityZ}) — " +
                          $"terrain={result.terrain} severity={result.severity} " +
                          $"ssim={result.ssim:F3} conf={result.confidence:F3}{replanTag}");
                break;

            case "complete":
                navigationComplete = true;
                Debug.Log($"RoverController: navigation complete — {result.message} " +
                          $"(replans={result.replans})");
                break;

            case "error":
                navigationComplete = true;
                Debug.LogError($"RoverController: server error — {result.message}");
                break;

            default:
                Debug.LogWarning($"RoverController: unexpected response type '{result.type}' — ignoring.");
                break;
        }
    }

    /// <summary>
    /// Measures how far this object's pivot (transform.position) sits above the
    /// lowest point of its own mesh geometry. If the rover's pivot isn't at its
    /// wheels — common after importing a model authored around a different origin —
    /// placing the pivot directly at the raycast hit makes it look like it's floating.
    /// This computes the real gap once, from the actual mesh, instead of a guessed number.
    /// </summary>
    void ComputePivotOffset()
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            pivotToBaseOffset = 0f;
            return;
        }

        Bounds combined = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            combined.Encapsulate(renderers[i].bounds);
        }

        pivotToBaseOffset = transform.position.y - combined.min.y;

        // Half-extents of the rover's actual footprint, used later to sample the
        // ground under its front/back/left/right rather than just its center.
        halfLength = Mathf.Max(combined.extents.z, 0.01f);
        halfWidth = Mathf.Max(combined.extents.x, 0.01f);

        Debug.Log($"RoverController: pivot sits {pivotToBaseOffset} units above the rover's lowest point — compensating automatically. " +
                  $"Footprint half-length={halfLength}, half-width={halfWidth}.");
    }

    void LoadPath()
    {
        string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);

        if (!File.Exists(path))
        {
            Debug.LogError($"RoverController: could not find {path}. " +
                            $"Make sure path_data.json is in Assets/StreamingAssets/.");
            return;
        }

        string json = File.ReadAllText(path);
        PathData data = JsonUtility.FromJson<PathData>(json);

        waypointsXZ.Clear();
        foreach (var wp in data.waypoints)
        {
            waypointsXZ.Add(new Vector2(wp.unity_x, wp.unity_z));
        }

        Debug.Log($"RoverController: loaded {waypointsXZ.Count} waypoints.");
    }

    void Update()
    {
        if (useNetworking)
        {
            UpdateNetworked();
        }
        else
        {
            UpdateFromStaticPath();
        }
    }

    void UpdateNetworked()
    {
        if (navigationComplete || waitingForServer || !currentTargetXZ.HasValue) return;

        Vector2 targetXZ = currentTargetXZ.Value;
        Vector3 currentPos = transform.position;
        Vector3 flatCurrent = new Vector3(currentPos.x, 0f, currentPos.z);
        Vector3 flatTarget = new Vector3(targetXZ.x, 0f, targetXZ.y);

        Vector3 nextFlat = Vector3.MoveTowards(flatCurrent, flatTarget, moveSpeed * Time.deltaTime);

        if (!MoveAndOrientTo(flatCurrent, nextFlat)) return; // footprint raycast missed — stay put

        if (Vector3.Distance(flatCurrent, flatTarget) <= arrivalThreshold)
        {
            // Reached this waypoint — ask the server for the next one, sending a fresh
            // camera frame with it. This blocks briefly (network round-trip); see
            // RoverNetworkClient's notes on why this is synchronous.
            RequestNextFromServer();
        }
    }

    void UpdateFromStaticPath()
    {
        if (waypointsXZ.Count == 0 || currentIndex >= waypointsXZ.Count) return;

        Vector2 targetXZ = waypointsXZ[currentIndex];
        Vector3 currentPos = transform.position;
        Vector3 flatCurrent = new Vector3(currentPos.x, 0f, currentPos.z);
        Vector3 flatTarget = new Vector3(targetXZ.x, 0f, targetXZ.y);

        Vector3 nextFlat = Vector3.MoveTowards(flatCurrent, flatTarget, moveSpeed * Time.deltaTime);

        if (!MoveAndOrientTo(flatCurrent, nextFlat)) return;

        if (Vector3.Distance(flatCurrent, flatTarget) <= arrivalThreshold)
        {
            currentIndex++;
            if (currentIndex >= waypointsXZ.Count && loop)
            {
                currentIndex = 0;
            }
        }
    }

    /// <summary>
    /// Shared movement step used by both the networked and static-path modes:
    /// samples the footprint at nextFlat, moves/tilts the rover there if the
    /// raycasts hit, and returns false (without moving) if they didn't.
    /// </summary>
    bool MoveAndOrientTo(Vector3 flatCurrent, Vector3 nextFlat)
    {
        Vector3 moveDir = nextFlat - flatCurrent;
        Vector3 forward = moveDir.sqrMagnitude > 0.0001f ? moveDir.normalized : transform.forward;
        Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;
        if (right.sqrMagnitude < 0.0001f) right = transform.right;

        bool allHit = SampleFootprint(nextFlat, forward, right,
            out Vector3 centerPoint, out Vector3 groundNormal);

        if (!allHit)
        {
            if (!loggedMissedRaycast)
            {
                Debug.LogWarning($"RoverController: a footprint raycast near ({nextFlat.x}, {nextFlat.z}) hit nothing — " +
                                  $"rover paused. Check the terrain's Mesh Collider (non-convex) and groundLayerMask.");
                loggedMissedRaycast = true;
            }
            return false;
        }

        transform.position = new Vector3(centerPoint.x, centerPoint.y + groundOffset + pivotToBaseOffset, centerPoint.z);

        if (alignToSlope)
        {
            Quaternion faceDir = Quaternion.LookRotation(
                Vector3.ProjectOnPlane(forward, groundNormal).normalized, groundNormal);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, faceDir, maxTurnDegreesPerSecond * Time.deltaTime);
        }

        return true;
    }

    /// <summary>
    /// Samples the ground at four points under the rover's actual footprint
    /// (front, back, left, right) instead of just its center. Returns the average
    /// height (so the body sits at a sensible mid-point rather than one corner)
    /// and a normal fitted through all four points (so pitch AND roll both follow
    /// the real slope, not just whichever single triangle the center happened to hit).
    /// </summary>
    bool SampleFootprint(Vector3 flatCenter, Vector3 forward, Vector3 right,
        out Vector3 centerPoint, out Vector3 normal)
    {
        float l = halfLength * footprintSampleFraction;
        float w = halfWidth * footprintSampleFraction;

        // 9-point grid (corners, edges, and center) instead of 4 edge points —
        // catches narrower terrain features than 4 points can, while still
        // producing height and tilt from the SAME fitted plane (see below), so
        // they can never disagree with each other the way the earlier "max height
        // with edge-only tilt" attempt did.
        float[] uOffsets = { -w, 0f, w };
        float[] vOffsets = { -l, 0f, l };

        List<Vector3> samples = new List<Vector3>(9); // x=u (right-offset), y=v (forward-offset), z=h (world height)
        foreach (float v in vOffsets)
        {
            foreach (float u in uOffsets)
            {
                float sx = flatCenter.x + right.x * u + forward.x * v;
                float sz = flatCenter.z + right.z * u + forward.z * v;
                if (!TryGetGround(sx, sz, out Vector3 hit, out _))
                {
                    centerPoint = default;
                    normal = Vector3.up;
                    return false;
                }
                samples.Add(new Vector3(u, v, hit.y));
            }
        }

        // Least-squares fit: h(u,v) = A*u + B*v + C, over all 9 samples.
        if (!SolvePlaneLeastSquares(samples, out float A, out float B, out float C))
        {
            centerPoint = default;
            normal = Vector3.up;
            return false;
        }

        // Height at the rover's own center comes directly from the fitted plane
        // (u=0, v=0 -> h=C) — the exact same plane used for tilt below, so the two
        // can never disagree the way separately-computed height/tilt did before.
        centerPoint = new Vector3(flatCenter.x, C, flatCenter.z);

        Vector3 tangentU = right + Vector3.up * A;
        Vector3 tangentV = forward + Vector3.up * B;
        normal = Vector3.Cross(tangentU, tangentV).normalized;
        if (Vector3.Dot(normal, Vector3.up) < 0f) normal = -normal; // keep it pointing up
        return true;
    }

    /// <summary>
    /// Least-squares fit of h = A*u + B*v + C through a set of (u, v, h) points
    /// (stored as Vector3.x=u, .y=v, .z=h), via the standard 3x3 normal-equations
    /// solve. Returns false only if the system is degenerate (shouldn't happen
    /// with the fixed 9-point grid used above, but guarded regardless).
    /// </summary>
    bool SolvePlaneLeastSquares(List<Vector3> points, out float A, out float B, out float C)
    {
        double Suu = 0, Suv = 0, Su = 0, Svv = 0, Sv = 0, N = points.Count;
        double Suh = 0, Svh = 0, Sh = 0;

        foreach (Vector3 p in points)
        {
            double u = p.x, v = p.y, h = p.z;
            Suu += u * u; Suv += u * v; Su += u;
            Svv += v * v; Sv += v;
            Suh += u * h; Svh += v * h; Sh += h;
        }

        // Solve the 3x3 system:
        // [Suu Suv Su] [A]   [Suh]
        // [Suv Svv Sv] [B] = [Svh]
        // [Su  Sv  N ] [C]   [Sh]
        double det = Suu * (Svv * N - Sv * Sv) - Suv * (Suv * N - Sv * Su) + Su * (Suv * Sv - Svv * Su);
        if (System.Math.Abs(det) < 1e-9)
        {
            A = B = C = 0f;
            return false;
        }

        double detA = Suh * (Svv * N - Sv * Sv) - Suv * (Svh * N - Sv * Sh) + Su * (Svh * Sv - Svv * Sh);
        double detB = Suu * (Svh * N - Sh * Sv) - Suh * (Suv * N - Sv * Su) + Su * (Suv * Sh - Svh * Su);
        double detC = Suu * (Svv * Sh - Sv * Svh) - Suv * (Suv * Sh - Sv * Suh) + Suh * (Suv * Sv - Svv * Su);

        A = (float)(detA / det);
        B = (float)(detB / det);
        C = (float)(detC / det);
        return true;
    }

    /// <summary>
    /// Casts a ray straight down at (x, z). Returns the hit point and surface normal
    /// (used both for the rover's height and for tilting it to match the slope).
    /// </summary>
    bool TryGetGround(float x, float z, out Vector3 point, out Vector3 normal)
    {
        Vector3 rayOrigin = new Vector3(x, raycastStartHeight, z);
        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, raycastMaxDistance, groundLayerMask))
        {
            loggedMissedRaycast = false;
            point = hit.point;
            normal = hit.normal;
            return true;
        }

        point = default;
        normal = Vector3.up;
        return false;
    }

    void SnapToWaypoint(int index, bool instant)
    {
        if (index < 0 || index >= waypointsXZ.Count) return;
        SnapToPosition(waypointsXZ[index]);
    }

    /// <summary>
    /// Instantly places the rover at the given X/Z, grounded and oriented correctly,
    /// with no interpolated movement. Used for the very first waypoint in either mode,
    /// so the rover doesn't visibly drive from the origin to its actual start point.
    /// </summary>
    void SnapToPosition(Vector2 xz)
    {
        Vector3 flatPos = new Vector3(xz.x, 0f, xz.y);
        if (SampleFootprint(flatPos, transform.forward, transform.right, out Vector3 point, out Vector3 normal))
        {
            transform.position = new Vector3(point.x, point.y + groundOffset + pivotToBaseOffset, point.z);
        }
    }
}