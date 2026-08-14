using UnityEngine;

/// <summary>
/// Follows the rover from behind and above, at a distance derived from the rover's
/// own measured size rather than a guessed magic number — so this doesn't need
/// re-tuning every time the rover's position/scale/model changes.
/// </summary>
public class CameraFollow : MonoBehaviour
{
    public Transform target;

    [Header("Auto-sizing")]
    [Tooltip("If true, distance/height below are computed from the target's actual mesh bounds at Start.")]
    public bool autoSize = true;
    public float distanceMultiplier = 4f;  // how many rover-lengths behind
    public float heightMultiplier = 2.5f;  // how many rover-heights above

    [Header("Manual fallback (used if Auto Sizing is off, or before it's computed)")]
    public float manualDistanceBehind = 2000f;
    public float manualHeightAbove = 1000f;

    public float smoothSpeed = 5f;
    [Tooltip("How quickly the camera's sense of 'forward' catches up to the rover's actual heading. Much lower than smoothSpeed on purpose — the rover's grid-based path can flip direction sharply at almost every waypoint, and reacting to that instantly is what causes camera whiplash on turns.")]
    public float headingSmoothSpeed = 1.2f;
    [Tooltip("Hard cap on how fast the camera can turn, in degrees per second — guarantees no whiplash regardless of how sharply the route changes direction.")]
    public float maxTurnDegreesPerSecond = 45f;
    [Tooltip("How far above the rover's pivot to aim the camera (keeps it looking at the body, not the ground at the rover's feet).")]
    public float lookTargetHeight = 200f;

    private float distanceBehind;
    private float heightAbove;
    private float selfClearance = 500f; // buffer around the rover's own body the obstruction check skips
    private bool sized = false;
    private Vector3 smoothedForward = Vector3.forward;
    private bool headingInitialized = false;

    void Start()
    {
        if (autoSize && target != null)
        {
            ComputeSizeBasedOffsets();
        }
        else
        {
            distanceBehind = manualDistanceBehind;
            heightAbove = manualHeightAbove;
        }
    }

    void ComputeSizeBasedOffsets()
    {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            Debug.LogWarning("CameraFollow: target has no renderers — falling back to manual offset.");
            distanceBehind = manualDistanceBehind;
            heightAbove = manualHeightAbove;
            return;
        }

        Bounds combined = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            combined.Encapsulate(renderers[i].bounds);
        }

        // Use the largest horizontal extent (length or width, whichever is bigger)
        // so the camera backs off enough regardless of which way the rover is facing.
        float horizontalExtent = Mathf.Max(combined.size.x, combined.size.z);
        distanceBehind = horizontalExtent * distanceMultiplier;
        heightAbove = combined.size.y * heightMultiplier;
        
        selfClearance = combined.size.magnitude * 0.75f;
        sized = true;

        Debug.Log($"CameraFollow: measured rover bounds size={combined.size} — " +
                  $"using distanceBehind={distanceBehind}, heightAbove={heightAbove}.");
    }

    [Header("Terrain clearance")]
    [Tooltip("Camera won't be placed lower than this far above the terrain surface directly beneath its computed position.")]
    public float minClearanceAboveTerrain = 300f;
    public float terrainRaycastStartHeight = 10000f;
    public float terrainRaycastMaxDistance = 20000f;
    public LayerMask terrainLayerMask = ~0;

    void LateUpdate()
    {
        if (target == null) return;

        if (autoSize && !sized)
        {
            ComputeSizeBasedOffsets();
        }

        // Flatten the rover's forward direction to the XZ plane so the camera's height
        // offset doesn't change as the rover pitches/rolls on slopes 
        Vector3 flatForward = target.forward;
        flatForward.y = 0f;
        if (flatForward.sqrMagnitude < 0.0001f) flatForward = Vector3.forward;
        flatForward.Normalize();

        
        if (!headingInitialized)
        {
            smoothedForward = flatForward;
            headingInitialized = true;
        }
        else
        {
            smoothedForward = Vector3.Slerp(smoothedForward, flatForward, headingSmoothSpeed * Time.deltaTime);
        }

        Vector3 desiredPosition = target.position
            - smoothedForward * distanceBehind
            + Vector3.up * heightAbove;

        // Don't let the camera end up inside/below the actual terrain surface at that
        // spot — the rover's elevation range is large enough that a fixed height offset
        // from the rover isn't enough; check the real ground there and enforce clearance.
        Vector3 rayOrigin = new Vector3(desiredPosition.x, terrainRaycastStartHeight, desiredPosition.z);
        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, terrainRaycastMaxDistance, terrainLayerMask))
        {
            float minY = hit.point.y + minClearanceAboveTerrain;
            if (desiredPosition.y < minY)
            {
                desiredPosition.y = minY;
            }
        }

   
        Vector3 lookAtPoint = target.position + Vector3.up * lookTargetHeight;
        Vector3 toCamera = desiredPosition - lookAtPoint;
        float desiredDistance = toCamera.magnitude;
        if (desiredDistance > selfClearance)
        {
            Vector3 dirToCamera = toCamera / desiredDistance;
            Vector3 checkStart = lookAtPoint + dirToCamera * selfClearance;
            float checkDistance = desiredDistance - selfClearance;

            if (Physics.Raycast(checkStart, dirToCamera, out RaycastHit blockHit, checkDistance, terrainLayerMask))
            {
                float safeDistance = selfClearance + Mathf.Max(blockHit.distance - minClearanceAboveTerrain, minClearanceAboveTerrain);
                desiredPosition = lookAtPoint + dirToCamera * safeDistance;
            }
        }

        transform.position = Vector3.Lerp(transform.position, desiredPosition, smoothSpeed * Time.deltaTime);

        Quaternion desiredRotation = Quaternion.LookRotation((lookAtPoint - transform.position).normalized, Vector3.up);
        transform.rotation = Quaternion.RotateTowards(transform.rotation, desiredRotation, maxTurnDegreesPerSecond * Time.deltaTime);
    }
}