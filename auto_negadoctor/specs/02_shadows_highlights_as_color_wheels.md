One aspect of this feature is finding per-frame shadows and highlights points, normally sampled from dark and light neutral grey.
The current algorithm has mistakes in finding them and I want to mass-tune it by showing you the correct hand-picked colors.
You currently present way to add such correction in debug UI via adding manual overrides to shadow/highlight patches and tuning sizes of that rectangles.
This is sort of good and should stay but often this process becomes vague and uncertain etc and when tuning it's best for me to just quickly go through multiple similar hues showing live-privew for each of them and then say 'this corrected picture now looks good, save this as good example for tuning' rather than trying to select patches of shadows and highlights from the picture.

Let's analyze what characterizes the shadow/highlight patches. Essentially, they are two color space points. I want two color wheels for choosing them. Dragging mouse pointer on them should affect realtime preview of the inverted image using that override. A colorwheel encodes both hue and intensity.

Chosen overrides are saved to output just like patches and should be a part of all tests, fixtures etc.

I don't know how the two ways of specifying shadows/highlights colors (via patches or wheels) should coexist in saved data. Give me options. One may theorize that after choosing a color via wheel the program could find a patch in the real image matching it - but is it always possible? what happens if not?..