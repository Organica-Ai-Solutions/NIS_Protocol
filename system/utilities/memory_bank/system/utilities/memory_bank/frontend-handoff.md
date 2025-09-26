# Frontend Handoff (Babylon viz + Runner integration)

Key contracts
- Viz intents (strict JSON in fenced ```viz blocks): types: scene_spec | step | chart | overlay
- scene_spec fields: engine:"babylon"; physics.engine:"cannon-es"|"havok-v2"; optional physics_api:"v1"|"v2"; meshes (box|sphere|plane|ground); materials; constraints (hinge|slider|ball); units {length, mass, time}
- step: ops on mesh/camera/light; dt ∈ [0, 0.033]; impulses clamped ≤ 1e4

Frontend tasks
- Chat parser: detect ```viz blocks; JSON.parse; whitelist keys; ignore unknown
- vizBus: publishVizIntent(intent); subscribers on /babylon-3d and inline canvas
- /babylon-3d: init on scene_spec; update on step; overlay to Babylon GUI; chart to 2D panel
- Physics V1 now: CannonJSPlugin with cannon-es; if physics.engine=="havok-v2" and physics_api=="v2" then route to future Havok V2 path (stub OK)
- Constraints: map hinge/slider/ball to V1 joints where available (stub if not supported); keep safe defaults
- Safety: cap update rate 30 Hz; respect prefers-reduced-motion; handle WebGL fallback to 2D chart

Runner endpoints (http://localhost:8001)
- POST /execute tool=orchestrate_plan {scene, build}
- POST /execute tool=submit_build {scene, build} → preview/index.html
- GET  /preview/index.html (static preview)

Inline visualize flow
- Show "Visualize?" CTA on assistant bubble if a viz intent exists
- On accept: render InlineBabylonView (lazy, ssr:false) 480x270; controls: Expand → /babylon-3d, Reset, Queue steps
- Queue intents until canvas ready

Postman
- Use NIS_Protocol_v3_COMPLETE_Postman_Collection.json → Runner Tools group

Notes
- Clamp/material presets/units now supported; ignore if not needed
- Ignore any non-whitelisted fields; never eval code; JSON only
