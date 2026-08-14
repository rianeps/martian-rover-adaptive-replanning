using UnityEngine;

/// <summary>

/// This must run AFTER RoverController each frame, since it adds a
/// small offset on top of whatever position/rotation RoverController just set.

/// </summary>
public class RoverJostle : MonoBehaviour
{
    [Header("Bounce (vertical bob)")]
    public float bobAmplitude = 3f;      
    public float bobFrequency = 4f;      

    [Header("Wobble (subtle roll/pitch)")]
    public float wobbleAmplitude = 1.5f; // degrees
    public float wobbleFrequency = 3f;

    [Tooltip("Movement speed at which the bounce/wobble reaches full strength. Below this, it scales down smoothly rather than cutting off abruptly.")]
    public float fullEffectSpeed = 50f;

    private Vector3 lastPosition;
    private float timeAccumulator = 0f;

    void Start()
    {
        lastPosition = transform.position;
    }

    void LateUpdate()
    {
        float speed = Vector3.Distance(transform.position, lastPosition) / Time.deltaTime;
        lastPosition = transform.position;

        float strength = Mathf.Clamp01(speed / Mathf.Max(fullEffectSpeed, 0.01f));
        if (strength < 0.001f) return; // stationary — don't jitter while parked

        timeAccumulator += Time.deltaTime;

        float bob = Mathf.Sin(timeAccumulator * bobFrequency * Mathf.PI * 2f) * bobAmplitude * strength;
        float rollWobble = Mathf.Sin(timeAccumulator * wobbleFrequency * Mathf.PI * 2f) * wobbleAmplitude * strength;
        float pitchWobble = Mathf.Cos(timeAccumulator * wobbleFrequency * 1.3f * Mathf.PI * 2f) * wobbleAmplitude * 0.6f * strength;

        transform.position += transform.up * bob;
        transform.rotation *= Quaternion.Euler(pitchWobble, 0f, rollWobble);
    }
}