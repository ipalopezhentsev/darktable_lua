Current implementation in detect_dust.py assumes a dust particle is a thing that is roughly circular and small and can be covered by a small single circular brush touch. In reality, not all dust is like this, some has different structure, it's like a thin hair or a thread going through a significant area of an image. We need to extend the existing algorithm to also detect and fix this kind of dust with brushes that are not single touch but a collection of brush strokes along the detected dust that repeat its path.

Examples:
1. Feather or thread dust: see `auto_retouch\specs\feather_dust.png`. There we see blue skies with a complex shaped thin fully white dust thread, it was fully opaque to scanning lite so was captured fully black and when the negative was inverted, forming images that we analyse, it turned to fully white object. 
2. Film scratch: see `auto_retouch\specs\film_scratch.png`. Has a bit different nature in that it's embedded into film and not simply laying on its surface, but the end result is roughly the same, elongated line-like whiteish line quite visible on blue skies.

Requirements:

1. Modify detection:
1.1. Introduce a new parameter enabling the feature (on by default)
1.2. You still have to continue detecting small round dust as now. But extend it: a) some of the dust that was kind of irregular shaped before and was retouched with a circular brush of large diameter now may be fixed by a brush line stroke along the dust. I.e. you have to weight what is more beneficial. b) start detecting two new kind of 'dust' explained above in examples and realize it would never be picked up by the existing logic, so something new is needed in addition.
2. Modify healing:
2.1. You have to heal the new types of dust not with a single brush touch but with a poly-line or spline going through the important points along the path of the detected dust/scratch.
2.2. Like in existing algorithm, it's vital to care about detecting source for the healing, near the detected dust, but not intersecting its area and similar to the background of the area where the dust was detected.
3. Modify debug UI:
3.1. Has to incorporate viewing/editing new detected brushes.
3.2. Such brushes have to be shown as lines/spline going through the key points and allowing to move the key points correcting them in annotation.
4. Modify regression testing scripts/baseline generator to incorporate newly added objects.
