import time
from pysiril.siril import Siril
from pysiril.wrapper import Wrapper

app = Siril(delai_start=1, bStable=False)
time.sleep(1)

cmd = Wrapper(app)
app.Open()
time.sleep(1)

cmd.set32bits()
cmd.setext("fit")
time.sleep(1)


light_dir = "/Users/gshau/astronomy/planner_data/demo/data/hickson_44/2021-04-02/"
out_dir = "/Users/gshau/astronomy/planner_data/demo/data/hickson_44/"

cmd.cd(light_dir)
cmd.convert("light", out=out_dir, fitseq=True)
cmd.cd(out_dir)
cmd.stack(
    "light",
    type="rej",
    sigma_low=3,
    sigma_high=3,
    norm="addscale",
    output_norm=True,
    out=f"../stack",
)
