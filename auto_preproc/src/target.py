from collections import defaultdict
import yaml
import sys
import shutil
import hashlib
import glob
import os
from threading import local
import time

from astro_planner.globals import BINNING_COL, EXPOSURE_COL, FOCALLENGTH_COL
from astropy.io import fits

from .logger import log
from .fit_header import get_lights, match_light_with_calibration, FILE_TYPES
from pysiril.siril import Siril
from pysiril.wrapper import Wrapper


def get_hash_of_file_list(file_list):
    s = "+".join(sorted(file_list))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[-6:]


def create_symlink(src, dst):
    if os.name == "nt":
        pass
    else:
        os.symlink(src, dst)


def status_wrap(success):
    if not success[0]:
        # log.warning("Issue!")
        raise Exception("Issue")


class ProcessTarget:
    def __init__(self, target_name, output_dir, master_cal_dir):
        self.target_name = target_name.replace(" ", "_")
        self.process_dir = f"{output_dir}/auto_process"
        self.master_cal_dir = master_cal_dir

        self.bias_dir = f"{output_dir}/linked_bias"
        self.flat_dir = f"{output_dir}/linked_flat"
        self.dark_dir = f"{output_dir}/linked_dark"
        self.light_dir = f"{output_dir}/linked_light"

        self.has_bias = len(glob.glob(f"{self.bias_dir}/*.FIT")) > 0
        self.has_flat = len(glob.glob(f"{self.flat_dir}/*.FIT")) > 0
        self.has_dark = len(glob.glob(f"{self.dark_dir}/*.FIT")) > 0
        self.has_light = len(glob.glob(f"{self.light_dir}/*.FIT")) > 0

        self.delay = 0.05

        self.bias_digest = None
        self.flat_digest = None
        self.dark_digest = None
        self.light_digest = None

        self.flat_file_list = {}
        self.bias_file_list = {}
        self.dark_file_list = {}

        os.makedirs(f"""{self.master_cal_dir}""", exist_ok=True)
        os.makedirs(f"""{self.process_dir}""", exist_ok=True)
        os.makedirs(f"""{self.bias_dir}""", exist_ok=True)
        os.makedirs(f"""{self.flat_dir}""", exist_ok=True)
        os.makedirs(f"""{self.dark_dir}""", exist_ok=True)
        os.makedirs(f"""{self.light_dir}""", exist_ok=True)
        # self.app = Siril(delai_start=1)
        try:
            if sys.platform.startswith("win32"):
                self.app = Siril(delai_start=1)
            else:
                self.app = Siril(
                    siril_exe="/Applications/SiriL.app/Contents/MacOS/siril-cli",
                    delai_start=1,
                    bStable=True,
                )
                time.sleep(self.delay)
        except:
            raise FileNotFoundError("Siril not found")
        self.cmd = Wrapper(self.app)
        self.app.Open()
        time.sleep(self.delay)
        self.cmd.set32bits()
        self.cmd.setext("fit")
        time.sleep(self.delay)

    def store_metadata(self, filename, detail_list):
        metadata_filename = filename.replace(".fit", ".txt")

        self.metadata = dict(
            [
                [k, v]
                for k, v in dict(self.__dict__).items()
                if k not in ["app", "cmd", "metadata"]
                and any([f"{m}_" in k for m in detail_list])
            ]
        )
        with open(metadata_filename, "w") as f:
            yaml.safe_dump(self.metadata, f)

    def helper(self, ftype, filter=None):
        not_cached = True
        r = dict(
            bias=self.bias_dir,
            dark=self.dark_dir,
            flat=self.flat_dir,
            light=self.light_dir,
        )

        log.info(ftype)
        log.info(r)
        file_list = sorted([os.path.basename(f) for f in glob.glob(f"{r[ftype]}/*")])
        log.info(file_list)
        digest = get_hash_of_file_list(file_list)
        master_file = f"{self.master_cal_dir}/{ftype}_{digest}_stacked.fit"
        local_master_file = f"{self.process_dir}/{ftype}_{digest}_stacked.fit"

        log.info(f"master_file: {master_file}")
        log.info(f"local_master_file: {local_master_file}")

        if filter is not None:
            master_file = f"{self.master_cal_dir}/{ftype}_{digest}_{filter}_stacked.fit"
            local_master_file = (
                f"{self.process_dir}/{ftype}_{digest}_{filter}_stacked.fit"
            )
        if os.path.exists(master_file):
            log.info(f"File exists: {master_file}")
            not_cached = False
            if not os.path.exists(local_master_file):
                log.info(f"Linking to file: {local_master_file}")
                create_symlink(master_file, local_master_file)
                create_symlink(
                    master_file.replace(".fit", ".txt"),
                    local_master_file.replace(".fit", ".txt"),
                )
        return master_file, local_master_file, digest, not_cached, file_list

    def process_bias(self):
        log.info("process_bias")
        if self.has_bias:
            master_file, local_master_file, digest, not_cached, file_list = self.helper(
                "bias"
            )
            self.bias_file_list = file_list
            self.bias_file_count = len(file_list)
            self.bias_digest = digest
            self.master_bias = f"bias_{digest}"
            if not_cached:
                self.cmd.cd(self.bias_dir)
                time.sleep(self.delay)
                self.cmd.convert(self.master_bias, out=self.process_dir, fitseq=True)
                time.sleep(self.delay)
                self.cmd.cd(self.process_dir)
                time.sleep(self.delay)
                self.cmd.stack(
                    self.master_bias, type="rej", sigma_low=3, sigma_high=3, norm="no"
                )
                time.sleep(self.delay)

                shutil.copy(local_master_file, master_file)
                self.store_metadata(local_master_file, detail_list=["bias"])
                self.store_metadata(master_file, detail_list=["bias"])
                os.unlink(f"{self.process_dir}/{self.master_bias}.fit")

    def process_dark(self):
        log.info("process_dark")
        if self.has_dark:
            master_file, local_master_file, digest, not_cached, file_list = self.helper(
                "dark"
            )
            self.dark_file_list = file_list
            self.dark_file_count = len(file_list)
            self.dark_digest = digest
            self.master_dark = f"dark_{digest}"
            if not_cached:
                self.cmd.cd(self.dark_dir)
                time.sleep(self.delay)
                self.cmd.convert(self.master_dark, out=self.process_dir, fitseq=True)
                time.sleep(self.delay)
                self.cmd.cd(self.process_dir)
                time.sleep(self.delay)
                self.cmd.stack(
                    self.master_dark,
                    type="rej",
                    sigma_low=3,
                    sigma_high=3,
                    norm="no",
                )
                time.sleep(self.delay)
                print("master_file: ", master_file)
                print("local_master_file: ", local_master_file)
                shutil.copy(local_master_file, master_file)
                self.store_metadata(local_master_file, detail_list=["dark"])
                self.store_metadata(master_file, detail_list=["dark"])
                os.unlink(f"{self.process_dir}/{self.master_dark}.fit")

    def process_flat(self, filter):
        log.info("process_flat")
        if self.has_flat:
            master_file, local_master_file, digest, not_cached, file_list = self.helper(
                "flat", filter
            )
            self.flat_file_list = file_list
            self.flat_file_count = len(file_list)
            self.flat_digest = digest
            self.master_flat = f"flat_{digest}_{filter}"
            if not_cached:
                self.cmd.cd(self.flat_dir)
                time.sleep(self.delay)
                self.cmd.convert(self.master_flat, out=self.process_dir, fitseq=True)
                time.sleep(self.delay)
                self.cmd.cd(self.process_dir)
                time.sleep(self.delay)
                preproc_kwargs = {}
                if self.has_bias:
                    preproc_kwargs["bias"] = f"{self.master_bias}_stacked"
                if self.has_dark:
                    preproc_kwargs["dark"] = f"{self.master_dark}_stacked"
                self.cmd.preprocess(self.master_flat, **preproc_kwargs)
                time.sleep(self.delay)
                self.cmd.stack(
                    f"pp_{self.master_flat}",
                    type="rej",
                    sigma_low=3,
                    sigma_high=3,
                    norm="mul",
                )
                time.sleep(self.delay)
                local_master_file = local_master_file.replace(
                    self.master_flat, f"pp_{self.master_flat}"
                )
                master_file = master_file.replace(
                    self.master_flat, f"pp_{self.master_flat}"
                )

                detail = ["flat"] + list(preproc_kwargs.keys())

                shutil.copy(local_master_file, master_file)
                self.store_metadata(local_master_file, detail_list=detail)
                self.store_metadata(master_file, detail_list=detail)
                os.unlink(f"{self.process_dir}/{self.master_flat}.fit")
                os.unlink(f"{self.process_dir}/pp_{self.master_flat}.fit")

    def process_light(
        self,
        target_name,
        filter,
        ref_frame=None,
        stack_kwargs={},
        n_lights=None,
        df_header=None,
    ):
        log.info("process_light")
        filename_out = ""
        if self.has_light:
            master_file, local_master_file, digest, not_cached, file_list = self.helper(
                "light", filter
            )
            self.light_digest = digest
            self.master_light = f"light_{digest}_{filter}"
            self.light_file_list = file_list
            self.light_file_count = len(file_list)
            log.info(f"moving to {self.light_dir}")
            status_wrap(self.cmd.cd(self.light_dir))
            time.sleep(self.delay)
            log.info(f"converting")
            status_wrap(
                self.cmd.convert(
                    f"light_{digest}_{filter}", out=self.process_dir, fitseq=True
                )
            )
            time.sleep(self.delay)
            log.info(f"moving to {self.process_dir}")
            status_wrap(self.cmd.cd(self.process_dir))
            time.sleep(self.delay)

            flat = None
            dark = None
            cfa = "OSC" == filter
            if self.has_flat:
                flat = f"{self.master_flat}_stacked"
                if self.has_bias or self.has_dark:
                    flat = f"pp_{flat}"
            if self.has_dark:
                dark = f"{self.master_dark}_stacked"
            log.info(f"preprocessing light_{digest}_{filter}")
            status_wrap(
                self.cmd.preprocess(
                    f"light_{digest}_{filter}",
                    dark=dark,
                    flat=flat,
                    cfa=cfa,
                    equalize_cfa=False,
                    debayer=cfa,
                )
            )
            time.sleep(self.delay)

            log.info(f"registering pp_light_{digest}_{filter}")
            status_wrap(self.cmd.register(f"pp_light_{digest}_{filter}"))
            time.sleep(self.delay)
            if ref_frame is not None:
                status_wrap(self.cmd.setref(f"pp_light_{digest}_{filter}", ref_frame))
                time.sleep(self.delay)

            df_lights = df_header[
                df_header["filename"]
                .apply(lambda f: os.path.basename(f))
                .isin(file_list)
            ]
            light_total_exposure = df_lights[EXPOSURE_COL].values.sum()
            self.light_total_exposure_min = int(light_total_exposure / 60)
            self.light_total_sub_count = df_lights[EXPOSURE_COL].shape[0]

            suffix = "_".join([f"{k}_{v}" for k, v in stack_kwargs.items()])
            suffix = f"{suffix}_{self.light_file_count}subs_{self.light_total_exposure_min}min"
            out_file = (
                f"{self.target_name}_master_{filter}_{digest}__TEMP__{suffix}".replace(
                    " ", "_"
                )
            )
            log.info(f"{out_file}, stack_kwargs: {stack_kwargs}")
            status_wrap(
                self.cmd.stack(
                    f"r_pp_light_{digest}_{filter}",
                    type="rej",
                    sigma_low=3,
                    sigma_high=3,
                    norm="addscale",
                    output_norm=True,
                    out=f"../{out_file}",
                    **stack_kwargs,
                )
            )

            log.info("fits open")
            filename_out = f"{self.process_dir}/../{out_file}.fit"
            log.info(f"Store metadata {filename_out}")
            with fits.open(filename_out, mode="update") as hdul:
                hdr = hdul[0].header
                hdr.set("OBJECT", target_name.rstrip().lstrip())
                hdr.set("FILTER", filter.rstrip().lstrip())
                # hdr.set(EXPOSURE_COL, hdr["EXPTIME"])
                hdr.set(EXPOSURE_COL, light_total_exposure)
                hdr.set("IMAGETYP", "MASTER LIGHT")
                hdr.set("AIP-CMT", "Processed with The AstroImaging Planner and Siril")
                hdul.flush()
            self.store_metadata(
                filename_out.replace("__TEMP__", ""),
                detail_list=["light", "flat", "dark", "bias"],
            )

            shutil.move(filename_out, filename_out.replace("__TEMP__", ""))

            log.info("unlink")
            os.unlink(f"{self.process_dir}/light_{digest}_{filter}.fit")
            os.unlink(f"{self.process_dir}/pp_light_{digest}_{filter}.fit")
            os.unlink(f"{self.process_dir}/r_pp_light_{digest}_{filter}.fit")

            self.cmd.close()
            log.info("Done")

        return filename_out


def process_target(
    target_name,
    lights_records,
    df_calibration,
    df_header,
    output_dir,
    master_cal_dir,
    stack_kwargs={},
    ref_frame_filter={},
    matching_files=None,
    preproc_list=[],
    app=None,
):
    output_dir = f"{output_dir}/{target_name}".replace(" ", "_")
    new_light_records = []
    n_steps = 6
    app.preproc_count = n_steps * len(lights_records)
    for i, lights_record in enumerate(lights_records):
        app.preproc_list.append(lights_record)
        app.preproc_progress += 1
        app.preproc_status = "Initializing"

        filter = lights_record.get("FILTER")
        binning = lights_record.get(BINNING_COL)
        exposure = lights_record.get(EXPOSURE_COL)
        nx = lights_record.get("NAXIS1")
        focal_length = lights_record.get(FOCALLENGTH_COL)

        extra_filter = {}
        for col in ["OFFSET", "GAIN"]:
            value = lights_record.get(col)
            if value is not None:
                extra_filter[col] = value

        dfh = df_header.copy()
        lights = get_lights(
            dfh,
            filter=filter,
            binning=binning,
            exposure=exposure,
            nx=nx,
            focal_length=focal_length,
            extra_filter=extra_filter,
        )
        if matching_files is not None:
            lights = [
                light for light in lights if os.path.basename(light) in matching_files
            ]
        calibration_files = match_light_with_calibration(lights_record, df_calibration)
        flats = calibration_files.get("flat", [])
        bias = calibration_files.get("bias", [])
        darks = calibration_files.get("dark", [])

        is_cfa = filter == "OSC"
        has_bias = len(bias) > 1
        has_darks = len(darks) > 1
        has_flats = len(flats) > 1
        has_lights = len(lights) > 1
        log.info(f"lights_record: {lights_record}")
        log.info(f"is_cfa: {is_cfa}")
        log.info(f"has_bias: {has_bias}")
        log.info(f"has_darks: {has_darks}")
        log.info(f"has_flats: {has_flats}")
        log.info(f"has_lights: {has_lights}")

        # SET DIRECTORIES
        for file_type in FILE_TYPES:
            dirname = os.path.join(output_dir, f"linked_{file_type}")
            os.makedirs(dirname, exist_ok=True)
            existing_files = glob.glob(f"{dirname}/*.FIT")
            for file in existing_files:
                if os.path.islink(file):
                    os.unlink(file)
                else:
                    log.info(f"file exists, not symlink: {file}")

        for file_type, file_set in zip(FILE_TYPES, [lights, darks, bias, flats]):
            for infile in file_set:
                infile_basename = os.path.basename(infile)
                outfile = os.path.join(
                    output_dir, f"linked_{file_type}", infile_basename
                )
                if os.path.exists(outfile):
                    continue
                create_symlink(infile, outfile)
        f = None
        try:
            f = ProcessTarget(target_name, output_dir, master_cal_dir)
            app.preproc_progress += 1
            app.preproc_status = "Processing Bias"
            f.process_bias()
            app.preproc_progress += 1
            app.preproc_status = "Processing Darks"
            f.process_dark()
            app.preproc_progress += 1
            app.preproc_status = "Processing Flats"
            f.process_flat(filter)
            app.preproc_progress += 1
            app.preproc_status = "Processing Lights"
            out_file = f.process_light(
                target_name,
                filter,
                ref_frame=ref_frame_filter.get(filter, None),
                stack_kwargs=stack_kwargs,
                df_header=dfh,
            )
            app.preproc_progress += 1
            app.preproc_status = "Done"
            lights_record["out_file"] = out_file
            new_light_records.append(lights_record)
        except KeyboardInterrupt:
            raise "Stopping"
        except:
            log.warning("Failed", exc_info=True)
        finally:
            if f is not None:
                f.cmd.close()
                f.app.Close()

        # for file_type in FILE_TYPES:
        #     dirname = f"{output_dir}/{file_type}"
        #     existing_files = glob.glob(f"{dirname}/*.FIT")
        #     for file in existing_files:
        #         if os.path.islink(file):
        #             os.unlink(file)
        #         else:
        #             log.info(f"file exists, not symlink: {file}")

    return new_light_records
