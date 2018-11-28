#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the code to load test the PYPI recommendation service.

Copyright © 2018 Red Hat Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from locust import HttpLocust, TaskSet, task
import json
import random


def get_packages(start, end):
    """Get random packages from list of packages."""
    packages = json.load(open('data/packages.json'))
    return random.sample(packages, random.randint(start, end))


class StackAnalysisUserBehaviour(TaskSet):
    """This class defines the user behaviours."""

    def on_start(self):
        """on_start is called when a Locust start before any task is scheduled."""
        pass

    @task
    def trigger_stack_analysis_package_stack(self):
        """Simulate a stack analysis request."""
        stack = [
            {
                "package_list": get_packages(1, 15),
            }
        ]
        self.client.post("/api/v1/companion_recommendation", data=json.dumps(stack),
                         headers={'Content-type': 'application/json'})


class StackAnalysisUserLocust(HttpLocust):
    """This class defines the params for the load testing piece."""

    min_wait = 10
    max_wait = 10
    task_set = StackAnalysisUserBehaviour
