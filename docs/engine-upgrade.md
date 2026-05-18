# Ada Core: Live2D & Model Control Upgrade Roadmap

## Phase 1: Continuous Parameter Control (Fluid Expressions)

Currently, the bridge relies on binary hotkey triggers (`*smiles*`). The next evolution is injecting raw floating-point values into Live2D parameters for granular, dynamic expressions.

* **LLM Sentiment Output:** Instruct the LLM to append a structured emotion array to its text (e.g., `[Joy: 0.8, Anger: 0.0]`).
* **VTS Parameter Injection:** Update `vts_bridge.py` to use the `InjectParameterData` API endpoint, feeding these values into custom VTS parameters (e.g., `Ada_Joy`).
* **Asynchronous Tweening:** Implement a mathematical lerp (linear interpolation) loop in the bridge's event loop so the avatar's face smoothly transitions to the target emotion over a few hundred milliseconds rather than snapping instantly.

## Phase 2: Dynamic Prop and Item Management

Expand the bridge to utilize the VTube Studio Item API, allowing the LLM to manipulate its physical environment.

* **Asset Spawning:** Map specific LLM actions (e.g., `*drinks coffee*`, `*puts on glasses*`) to the `ItemLoad` API request to spawn pre-configured PNG or Live2D item files into the scene.
* **ArtMesh Attachment:** Programmatically pin these items to specific points on the Live2D rig (like attaching a mug to the hand mesh or sunglasses to the head mesh) so they track perfectly with the avatar's physics and movement.
* **Lifecycle Management:** Implement automatic despawn timers or removal commands so the screen doesn't become cluttered with props over a long session.

## Phase 3: ArtMesh Tinting & Lighting Simulation

Use the VTS API to dynamically recolor parts of the model to reflect extreme emotions or changing scene lighting.

* **Emotional Tinting:** Map high emotion scores to the `ColorTint` API request. For example, tinting the cheek ArtMeshes red for embarrassment or dropping a dark, transparent shadow over the upper face meshes for anger or concentration.
* **Environmental Lighting:** Create Python functions that can wash the entire model in specific hex colors to simulate the glow of a computer monitor, alarm lights, or a specific time of day, entirely bypassing the need to rig separate lighting toggles inside Live2D.

## Phase 4: Spacial & Camera Direction

Give the AI control over its physical presence on the screen by connecting the bridge to VTube Studio's camera and positioning controls.

* **Dynamic Framing:** Allow the LLM to output spacial tags (`*leans in*`, `*backs away*`). The bridge intercepts these and sends `MoveModel` API requests to smoothly zoom the camera in on the face for quiet dialogue or zoom out for casual chat.
* **Screen Context Avoidance:** Program the bridge to automatically pan the avatar to the left or right side of the monitor based on system context, ensuring the model never blocks critical UI elements while idling or observing.