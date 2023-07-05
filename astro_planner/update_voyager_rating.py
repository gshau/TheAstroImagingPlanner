#!/usr/bin/env python3

import _thread
import asyncio
import base64
import hashlib
import json
import time
import uuid
import uuid
import requests


from collections import deque
from threading import Thread

import websocket

import os
import logging

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(module)s:%(lineno)d %(message)s",
)
log = logging.getLogger(__name__)


class VoyagerConnectionManager(Thread):
    """
    Low level class that maintains a live connection with voyager application server.
    It allows command sending, keep-alive, reconnect, etc.
    Logic to understand the content of each packet lives in 'VoyagerClient'.
    """

    def __init__(
        self,
        config=None,
        server_url="0.0.0.0",
        server_port=5950,
        thread_id: str = "WSThread",
    ):
        Thread.__init__(self)
        self.thread_id = thread_id
        self.wst = None

        self.server_url = server_url
        self.server_port = server_port

        self.config = config
        # self.voyager_settings = self.config.voyager_setting

        self.ws = None
        self.keep_alive_thread = None
        self.command_queue = deque([])
        self.ongoing_command = None
        self.current_command_future = None
        self.next_id = 1
        self.msg_list = []

        # self.log_writer = LogWriter(config=config)

        self.reconnect_delay_sec = 1
        self.should_exit_keep_alive_thread = False

        self.receive_message_callback = None
        self.bootstrap = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def send_command(self, command_name, params, with_mac=False, extra=""):
        # log.info(f"sending command..{command_name}")
        uid = str(uuid.uuid1())
        params["UID"] = uid
        if with_mac:
            mac_raw = f"secret99{uid}{extra}"
            # log.info(f"mac-raw: {mac_raw}")
            mac = hashlib.md5(mac_raw.encode("utf-8")).hexdigest()
            params["MAC"] = mac
            # log.info(f"mac: {mac}")

        command = {"method": command_name, "params": params, "id": self.next_id}
        self.next_id = self.next_id + 1
        event_loop = asyncio.get_event_loop()
        future = event_loop.create_future()
        # future = asyncio.Future()

        self.command_queue.append((command, future))
        self.try_to_process_next_command()
        log.info(
            f"about to wait for a future, related to command {command_name}, {future}"
        )
        while not future.done():
            await asyncio.sleep(1)
        await future
        result = future.result()
        log.info(f"a previous future was complete, result is: {result}, {future}")
        return result

    def try_to_process_next_command(self):
        if self.ongoing_command is not None:
            log.info("wait a while before sending out the second command.")
            # this command will be invoked later for sure
            return

        if len(self.command_queue) == 0:
            return

        command, future = self.command_queue.popleft()
        log.info(f"Trying to send out command {command}")
        self.ongoing_command = command
        self.current_command_future = future
        self.ws.send(json.dumps(command) + "\r\n")
        log.info("Sending command done")

    def on_message(self, ws, message_string):
        if not message_string or not message_string.strip():
            # Empty message string, nothing to do
            return

        message = json.loads(message_string)
        # log.info(f"*** Message string ***: {message_string}")
        # self.log_writer.write_line(message_string)

        if "jsonrpc" in message:
            log.info("received a message that looks like a method result")
            # some command finished, try to see if we have anything else.
            self.ongoing_command = None
            log.info(
                f"setting future result to {message}, {self.current_command_future}"
            )
            self.current_command_future.set_result(message)
            log.info(f"setting future result done {self.current_command_future}")
            self.try_to_process_next_command()
            return

        event = message["Event"]
        # self.voyager_client.parse_message(event, message)
        if self.receive_message_callback:
            response = self.receive_message_callback(event, message)
            if response is not None:
                self.msg_list = response

    def on_error(self, ws, error):
        # self.log_writer.maybe_flush()
        log.info("### {error} ###".format(error=error))

    def on_close(self, ws, close_status_code, close_msg):
        log.info("### [{code}] {msg} ###".format(code=close_status_code, msg=close_msg))
        # try to reconnect with an exponentially increasing delay
        self.should_exit_keep_alive_thread = True
        self.keep_alive_thread = None

        # if True:
        #     time.sleep(self.reconnect_delay_sec)
        #     if self.reconnect_delay_sec < 512:
        #         # doubles the reconnect delay so that we don't DOS server.
        #         self.reconnect_delay_sec = self.reconnect_delay_sec * 2
        #     # reset keep alive thread
        #     self.should_exit_keep_alive_thread = True
        #     self.keep_alive_thread = None
        #     self.run_forever()

    def on_open(self, ws):
        # Reset the reconnection delay to 1 sec
        # if self.bootstrap:
        #    task = self.loop.create_task(self.bootstrap())
        #    self.loop.run_until_complete(task)

        self.reconnect_delay_sec = 1

        if self.keep_alive_thread is None:
            self.should_exit_keep_alive_thread = False
            self.keep_alive_thread = _thread.start_new_thread(
                self.keep_alive_routine, ()
            )

    def run_forever(self):
        log.info(f"{self.thread_id} Starting thread")

        self.ws = websocket.WebSocketApp(
            "ws://{server_url}:{port}/".format(
                server_url=self.server_url, port=self.server_port
            ),
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.keep_running = True
        self.wst = Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()

    def keep_alive_routine(self):
        while not self.should_exit_keep_alive_thread:
            self.ws.send(
                '{"Event":"Polling","Timestamp":%d,"Inst":1}\r\n' % time.time()
            )
            time.sleep(5)


def receive_message_callback(e, m):
    if e == "RemoteActionResult":
        # log.info(e, m)
        l = m.get("ParamRet").get("list")
        return l


def get_attr_from_list(list_arg, search_key, search_value, attr_name):
    for dict in list_arg:
        if dict.get(search_key) == search_value:
            attr_value = dict.get(attr_name)
            return attr_value


async def main(
    auth_token,
    target_names=None,
    df_ratings=None,
    server_url="localhost",
    server_port=5950,
):

    assert target_names is None or df_ratings is None

    websocket.enableTrace(False)
    connection_manager = VoyagerConnectionManager(
        server_url=server_url, server_port=server_port
    )
    connection_manager.run_forever()
    connection_manager.receive_message_callback = receive_message_callback

    time.sleep(1)

    encoded_token = base64.urlsafe_b64encode(auth_token.encode("ascii"))
    result = await connection_manager.send_command(
        "AuthenticateUserBase", {"Base": encoded_token.decode("ascii")}
    )
    # log.info(f"RESULT 0: {result}")

    time.sleep(0.2)

    # Get targets
    params = {}
    result = await connection_manager.send_command(
        "RemoteOpenRoboTargetGetTargetList", params, with_mac=True
    )
    target_list = connection_manager.msg_list

    shot_list = []

    if target_names is None:
        target_names = df_ratings["OBJECT"].unique()

    for target_name in target_names:
        ref_guid_target = get_attr_from_list(
            target_list, "targetname", target_name, "guid"
        )
        log.info(f"ref_guid_target: {ref_guid_target}")

        # Get shots under target
        params = {}
        params["RefGuidTarget"] = ref_guid_target
        params["IsDeleted"] = False
        result = await connection_manager.send_command(
            "RemoteOpenRoboTargetGetShotDoneList",
            params,
            with_mac=True,
            extra=params["RefGuidTarget"],
        )

        shot_list += connection_manager.msg_list
        log.info(shot_list)

        if df_ratings is None:
            r = requests.get(f"http://127.0.0.1:5678/status/{target_name}")
            text = r.text
            ratings = json.loads(text)
        else:
            ratings = json.dumps(
                df_ratings.to_dict(orient="records"),
                sort_keys=True,
                indent=4,
                separators=(",", ": "),
            )

        filename_to_guid = {}
        for s in shot_list:
            filename_to_guid[s.get("filename")] = s.get("guid")

        src_list = []
        for rating in ratings:
            filename = rating.get("filename")
            rating = rating.get("is_ok")
            guid = filename_to_guid.get(filename)
            if guid is None:
                continue
            result = dict(RefGuidShotDone=guid, Rating=rating, IsToDelete=rating == 0)
            src_list.append(result)

        time.sleep(0.2)

        params = {}
        params["ObjUID"] = ref_guid_target
        params["Mode"] = 2
        params["RatingMode"] = 0
        params["RatingLimit"] = 0
        result = await connection_manager.send_command(
            "RemoteOpenRoboTargetRestoreShotDone",
            params,
            with_mac=True,
            extra=params["ObjUID"],
        )

        time.sleep(0.2)

        params = {}
        params["SrcList"] = src_list
        params["isDeleted"] = False
        result = await connection_manager.send_command(
            "RemoteOpenRoboTargetUpdateBulkShotDone",
            params,
            with_mac=True,
        )

    connection_manager.ws.close()
    log.info("Done!")
