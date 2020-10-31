from astro_planner.target import DEC_KEY, RA_KEY, TARGET_KEY, DATA_DIR_KEY, PROFILE_KEY
from astro_planner.profile import DEFAULT_DATA_PATH, DEFAULT_PROFILE_PATH

VOYAGER_SEQUENCE_EXTENSION = "s2q"
VOYAGER_PROFILE_EXTENSION = "v2y"


class VoyagerSequenceWriter:
    def __init__(self, sequence, sequence_template="./templates/VOYAGER_TEMPLATE.s2q"):
        self.sequence = sequence
        self.profile = sequence.profile
        self.target = sequence.target
        self.sequence_template = sequence_template
        self.n_rounds = sequence.n_rounds
        self.item_template = (
            '<item id="ref-{ref_id}" xsi:type="SOAP-ENC:string">{value}</item>'
        )

        self.filter_sequence = sequence.filter_sequence
        self.n_slots = len(self.filter_sequence.keys())
        self.gain_refs = [132 + i + 1 for i in range(self.n_slots)]
        self.el_refs = [self.gain_refs[-1] + 1 + i for i in range(self.n_slots)]

        self._read_template()
        self._update_lines()
        self.add_filters()

    def _read_template(self):
        with open(self.sequence_template, "r") as f:
            self.sequence_lines = f.readlines()

    def _update_lines(self):

        target_dict = {
            "125": RA_KEY,
            "127": DEC_KEY,
            "129": TARGET_KEY,
            "128": DATA_DIR_KEY,
            "130": PROFILE_KEY,
        }

        target_name = self.target.name.lower().replace(" ", "_")
        data_dir = f"{DEFAULT_DATA_PATH}\{target_name}"

        profile_filename = (
            f"{DEFAULT_PROFILE_PATH}\{self.profile.name}.{VOYAGER_PROFILE_EXTENSION}"
        )

        value_dict = {
            RA_KEY: self.target.ra_string,
            DEC_KEY: self.target.dec_string,
            TARGET_KEY: self.target.name,
            DATA_DIR_KEY: data_dir,
            PROFILE_KEY: profile_filename,
        }

        for i_line, line in enumerate(self.sequence_lines):
            for ref_id in target_dict.keys():
                if "ref-{id}".format(id=ref_id) in line:
                    output = self.item_template.format(
                        ref_id=ref_id, value=value_dict[target_dict[ref_id]]
                    )
                    self.sequence_lines[i_line] = output

    def _preq_filter(self):

        string_list = []
        string_list.append(
            """<a1:ArrayList id="ref-124" xmlns:a1="http://schemas.microsoft.com/clr/ns/System.Collections">\n<_items href="#ref-131"/>\n<_size>{n_slots}</_size>\n<_version>26</_version>\n</a1:ArrayList>\n<a1:ArrayList id="ref-126" xmlns:a1="http://schemas.microsoft.com/clr/ns/System.Collections">\n<_items href="#ref-132"/>\n<_size>{n_slots}</_size>\n<_version>169</_version>\n</a1:ArrayList>""".format(
                n_slots=self.n_slots
            )
        )
        string_list.append(
            """<SOAP-ENC:Array id="ref-131" SOAP-ENC:arrayType="xsd:anyType[8]">"""
        )
        for i_ref in self.gain_refs:
            string_list.append('<item href="#ref-{}"/>'.format(i_ref))
        string_list.append(
            """</SOAP-ENC:Array>\n<SOAP-ENC:Array id="ref-132" SOAP-ENC:arrayType="xsd:anyType[8]">"""
        )
        for i_ref in self.el_refs:
            string_list.append('<item href="#ref-{}"/>'.format(i_ref))
        string_list.append("""</SOAP-ENC:Array>""")
        return string_list

    def _gain_offset(self, ref_id, slot_number):
        string = """<a3:SequenzaElementoGainOffset id="ref-{ref_id}" xmlns:a3="http://schemas.microsoft.com/clr/nsassem/Voyager2/Voyager2%2C%20Version%3D1.0.0.0%2C%20Culture%3Dneutral%2C%20PublicKeyToken%3Dnull">\n<SlotNumber>{slot_number}</SlotNumber>\n<Gain>0</Gain>\n<Offset>0</Offset>\n</a3:SequenzaElementoGainOffset>""".format(
            ref_id=ref_id, slot_number=slot_number
        )
        return string

    def _seq_element(
        self, ref_id, filter_pos, filter_name, exposure, binning, n_exposure
    ):

        string = """<a3:SequenzaElemento id="ref-{ref_id}" xmlns:a3="http://schemas.microsoft.com/clr/nsassem/Voyager2/Voyager2%2C%20Version%3D1.0.0.0%2C%20Culture%3Dneutral%2C%20PublicKeyToken%3Dnull">\n<mTipoEsposizione>0</mTipoEsposizione>\n<mFiltroIndice>{filter_pos}</mFiltroIndice>\n<mFiltroLabel href="#ref-123"/>\n<mEsposizioneSecondi>{exposure}</mEsposizioneSecondi>\n<mBinning>{binning}</mBinning>\n<mNumero>{n_exposure}</mNumero>\n<mSpeedIndice>0</mSpeedIndice>\n<mReadoutIndice>1</mReadoutIndice>\n<mPlanningHelpElaborate>0</mPlanningHelpElaborate>\n<mEseguite>0</mEseguite>\n<mStatisticheSub xsi:null="1"/>\n<mLastInsNum>0</mLastInsNum>\n</a3:SequenzaElemento>""".format(
            ref_id=ref_id,
            filter_pos=filter_pos,
            ref_idp1=ref_id + 1,
            filter_name=filter_name,
            exposure=exposure,
            binning=binning,
            n_exposure=n_exposure,
        )

        return string

    def add_filters(self):

        preq_list = self._preq_filter()

        id_ref = 132
        filter_list = []
        for slot_number, (filter_name, data) in enumerate(self.filter_sequence.items()):
            id_ref += 1
            string = self._gain_offset(self.gain_refs[slot_number], slot_number + 1)
            filter_list.append(string)

        for slot_number, (filter_name, data) in enumerate(self.filter_sequence.items()):
            id_ref += 1
            string = self._seq_element(
                self.el_refs[slot_number],
                self.sequence.profile.sensor.filter_wheel.filter_pos[filter_name],
                filter_name,
                data["exposure"],
                data["binning"],
                data["n_subs"],
            )
            filter_list.append(string)

        self.filter_lines = preq_list + filter_list

    def write_file(self, file_out="test.s2q"):
        FOOTER = "</SOAP-ENV:Body>\n</SOAP-ENV:Envelope>"
        lines_out = [
            line.replace("\n", "")
            for line in self.sequence_lines
            if "</SOAP-ENV" not in line
        ]
        lines = lines_out + self.filter_lines + [FOOTER]
        with open(file_out, "w") as file_out:
            file_out.write("\n".join(lines))
