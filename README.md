# DEOE
## Framework
![Framework](./git_figs/framework.png)
## Abstract
Object detection is critical in autonomous driving, and it is practically demanding yet more challenging to localize objects
of unknown categories, known as Class-Agnostic Object Detection (CAOD). Existing studies on CAOD predominantly rely on ordinary
cameras, but these frame-based sensors are sensitive to motion and illumination, leading to safety risks in real-world scenarios. In
this study, we turn to a new modality enabled by the so-called event camera, featured by its sub-millisecond latency and high dynamic
range, for robust CAOD. Introducing Detecting Every Object in Events (DEOE), our approach can achieve high-speed, class-agnostic,
and open-world object detection for event-based vision. Built upon the fast event-based backbone: recurrent vision transformer, we
jointly consider the spatial and temporal consistencies to identify potential objects. The discovered potential objects are assimilated as
soft positive samples to avoid being suppressed as background. In addition, we disentangle the foreground-background classification
and novel object discovery task via a disentangled objectness head, enhancing the generalization of localizing novel objects while
maintaining the strong ability to filter out the background. Extensive experiments confirm the superiority of our proposed DEOE in
comparison with three strong baseline methods that integrate the state-of-the-art event-based object detector with advancements in
frame-based CAOD.

![Open class: bicycle](.gifs/bicycle.mp4)
