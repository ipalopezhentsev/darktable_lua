detect_dust.py finds dust-like points on input images and either dumps them to file for human analysis or to special format later read by auto_retouch.lua which transfers them to Darktable's XMP format for final result, that is a photo in Darktable with a new editing item in history of actions: retouched spots. I.e. the goal is to remove most visible dust particles.

if the same is done by a human in Darktable's UI, there, in addition to simple clicking on spots we want to retouch, we have several additional controls for each brush point:
- which source Darktable should use to inpaint the detected spot (we use "healing tool")
- brush size
- brush hardness
- brush opacity

Of these, the first one is critical to get right. By default darktable creates a source point which is a bit down and a bit to the right from the found dust particle. I.e. it will look at what's at this source point and try to paint the found dust particle with similar stuff. 

But this default location is not always the right one. Say a dust particle was in the sky and default healing source happens to land on some busy detail like a building - then sky will be painted with wrong content.

So the goal of this feature is to enrich the current detection algorithm with also finding the correct source point.
It should try to look around the found dust particle and find best candidate for healing source:
- close by color
- close by texture
- not being another dust particle
- not very far from the original point
- not being within detected brush size
- something else?

The feature should be added holistically:
- to detection algo
- detected sources should added to the xmp, so the lua part
- to debug output
- to debug UI (should allow repositioning found source to the "right" place for human)
- to test harness/baselines

