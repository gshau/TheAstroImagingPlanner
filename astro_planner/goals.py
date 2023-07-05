import pandas as pd
import numpy as np


class FilterGoal:
    def __init__(
        self,
        filter_name,
        total_exposure,
        binning=1,
        sub_exposure=300,
        acquired_exposure=0,
    ):
        self.filter_name = filter_name
        self.binning = binning
        self.sub_exposure = sub_exposure
        self.total_exposure = total_exposure
        self.acquired_exposure = acquired_exposure
        self.update_progress(acquired_exposure)

    def update_goal(self, **kwargs):
        if "binning" in kwargs:
            self.binning = kwargs.get("binning")
        if "sub_exposure" in kwargs:
            self.sub_exposure = kwargs.get("sub_exposure")
        if "total_exposure" in kwargs:
            self.total_exposure = kwargs.get("total_exposure")

    def update_progress(self, acquired_exposure):
        self.acquired_exposure = acquired_exposure

    @property
    def progress(self):
        return self.acquired_exposure / self.total_exposure

    @property
    def sub_count(self):
        return int(np.ceil(self.total_exposure / self.sub_exposure))

    @property
    def df(self):
        return pd.DataFrame.from_records([self.__dict__]).set_index("filter_name")

    @property
    def is_complete(self):
        return self.acquired_exposure > self.total_exposure

    def serialize(self):
        return self.__dict__

    @classmethod
    def deserialize(cls, record):
        assert "filter_name" in record
        assert "total_exposure" in record
        return cls(**record)

    def __repr__(self):
        return f"FilterGoal for {self.filter_name}: {self.sub_count}x{self.sub_exposure}s, {self.total_exposure / 60} min total"


class ImageGoal:
    def __init__(self, filter_goals=None):
        self.filter_goals = filter_goals

    def update_goal(self, filter_name, filter_goal):
        self.update_goals({filter_name: filter_goal})

    def update_goals(self, filter_goals):
        self.filter_goals.update(filter_goals)

    def update_progress(self, filter_name, exposure):
        if filter_name in self.filter_goals.keys():
            self.filter_goals[filter_name].update_progress(exposure)

    @property
    def progress(self):
        goal_progress = {}
        for filter_name, filter_goal in self.filter_goals.items():
            goal_progress[filter_name] = filter_goal.progress
        return goal_progress

    @property
    def total_exposure(self):
        goal_progress = {}
        for filter_name, filter_goal in self.filter_goals.items():
            goal_progress[filter_name] = filter_goal.acquired_exposure
        return goal_progress

    @property
    def total_requested_exposure(self):
        goal_progress = {}
        for filter_name, filter_goal in self.filter_goals.items():
            goal_progress[filter_name] = filter_goal.total_exposure
        return goal_progress

    @property
    def df(self):
        df_list = []
        for filter_name, filter_goal in self.filter_goals.items():
            df_list.append(filter_goal.df)
        return pd.concat(df_list)

    def serialize(self):
        return self.df.reset_index().to_dict(orient="records")

    @classmethod
    def deserialize(cls, records):
        goals = [FilterGoal(**record) for record in records]
        filter_names = [goal.filter_name for goal in goals]
        filter_goals = dict(zip(filter_names, goals))
        return cls(filter_goals)

    def __repr__(self):
        return f"Image Goal {self.filter_goals}"
