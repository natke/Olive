# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_pytorch_model
from unittest.mock import MagicMock, patch

import pytest

from olive.evaluator.metric import AccuracySubType, joint_metric_key
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.systems.common import LocalDockerConfig
from olive.systems.docker.docker_system import DockerSystem


class TestDockerSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    @patch("olive.systems.docker.docker_system.docker.from_env")
    def test__init_image_exist(self, mock_from_env):
        # setup
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        mock_image = MagicMock()
        mock_docker_client.images.get.return_value = mock_image
        docker_config = LocalDockerConfig(
            image_name="image_name", build_context_path="build_context_path", dockerfile="dockerfile"
        )

        # execute
        docker_system = DockerSystem(docker_config, is_dev=True)

        # assert
        assert docker_system.image == mock_image
        mock_docker_client.images.get.called_once_with(docker_config.image_name)

    @patch("olive.systems.docker.docker_system.docker.from_env")
    def test__init_image_dockerfile_build(self, mock_from_env):
        # setup
        import docker

        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        docker_config = LocalDockerConfig(
            image_name="image_name", build_context_path="build_context_path", dockerfile="dockerfile"
        )
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("msg")
        mock_docker_client.images.build = MagicMock()

        # execute
        DockerSystem(docker_config, is_dev=True)

        # assert
        mock_docker_client.images.build.called_once_with(
            docker_config.build_context_path,
            docker_config.dockerfile,
            docker_config.image_name,
            docker_config.build_args,
        )

    @patch("olive.systems.docker.docker_system.shutil.copy2")
    @patch("olive.systems.docker.docker_system.docker.from_env")
    @patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory")
    def test__init_image_requirements_file_build(self, mock_tempdir, mock_from_env, mock_copy):
        # setup
        import docker

        tempdir = self.tmp_dir.name
        mock_tempdir.return_value.__enter__.return_value = tempdir
        mock_docker_client = MagicMock()
        mock_from_env.return_value = mock_docker_client
        docker_config = LocalDockerConfig(
            image_name="image_name",
            requirements_file_path="requirements_file_path",
        )
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("msg")
        mock_docker_client.images.build = MagicMock()

        # execute
        docker_system = DockerSystem(docker_config, is_dev=True)

        # assert
        mock_docker_client.images.build.called_once_with(
            tempdir, docker_system.BASE_DOCKERFILE, docker_config.image_name, docker_config.build_args
        )

    @pytest.fixture
    def mock_docker_system_info(self):
        self.mock_from_env = patch("olive.systems.docker.docker_system.docker.from_env").start()
        self.mock_tempdir = patch("olive.systems.docker.docker_system.tempfile.TemporaryDirectory").start()
        self.mock_create_eval_script_mount = patch(
            "olive.systems.docker.docker_system.docker_utils.create_eval_script_mount"
        ).start()
        self.mock_create_model_mount = patch(
            "olive.systems.docker.docker_system.docker_utils.create_model_mount"
        ).start()
        self.mock_create_metric_volumes_list = patch(
            "olive.systems.docker.docker_system.docker_utils.create_metric_volumes_list"
        ).start()
        self.mock_create_config_file = patch(
            "olive.systems.docker.docker_system.docker_utils.create_config_file"
        ).start()
        self.mock_create_output_mount = patch(
            "olive.systems.docker.docker_system.docker_utils.create_output_mount"
        ).start()
        self.mock_create_evaluate_command = patch(
            "olive.systems.docker.docker_system.docker_utils.create_evaluate_command"
        ).start()
        self.mock_create_run_command = patch(
            "olive.systems.docker.docker_system.docker_utils.create_run_command"
        ).start()
        self.mock_copy = patch("olive.systems.docker.docker_system.copy.deepcopy").start()
        yield
        patch.stopall()

    @pytest.mark.usefixtures("mock_docker_system_info")
    @pytest.mark.parametrize("exit_code", [0, 1])
    def test_evaluate_model(
        self,
        exit_code,
    ):
        # setup
        import docker

        mock_docker_client = MagicMock()
        self.mock_from_env.return_value = mock_docker_client
        self.mock_from_env.return_value.containers.run.return_value.wait.return_value = {"StatusCode": exit_code}
        if exit_code != 0:
            self.mock_from_env.return_value.containers.run.return_value.logs.return_value = [b"mock_error"]
        tempdir = self.tmp_dir.name
        self.mock_tempdir.return_value.__enter__.return_value = tempdir
        olive_model = get_pytorch_model()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        docker_config = LocalDockerConfig(
            image_name="image_name", build_context_path="build_context_path", dockerfile="dockerfile"
        )
        docker_system = DockerSystem(docker_config, is_dev=True)
        container_root_path = Path("/olive/")
        eval_output_path = "eval_output"
        eval_output_name = "eval_res.json"
        mock_copy = MagicMock()
        mock_copy.return_value = mock_copy

        eval_file_mount_path = "eval_file_mount_path"
        eval_file_mount_str = "eval_file_mount_str"
        self.mock_create_eval_script_mount.return_value = [eval_file_mount_path, eval_file_mount_str]

        model_mount_path = "model_mount_path"
        model_mount_str_list = ["model_mount_str_list"]
        self.mock_create_model_mount.return_value = [model_mount_path, model_mount_str_list]

        volumes_list = ["volumes_list"]
        self.mock_create_metric_volumes_list.return_value = volumes_list

        config_mount_path = "config_mount_path"
        config_file_mount_str = "config_file_mount_str"
        self.mock_create_config_file.return_value = [config_mount_path, config_file_mount_str]

        output_local_path = Path(__file__).absolute().parent / "output_local_path"
        output_mount_path = "output_mount_path"
        output_mount_str = "output_mount_str"
        self.mock_create_output_mount.return_value = [output_local_path, output_mount_path, output_mount_str]

        eval_command = "eval_command"
        self.mock_create_evaluate_command.return_value = eval_command

        run_command = {"key": "val"}
        self.mock_create_run_command.return_value = run_command

        if exit_code != 0:
            with pytest.raises(
                docker.errors.ContainerError,
                match=r".*returned non-zero exit status 1: Docker container evaluation failed with: mock_error",
            ):
                actual_res = docker_system.evaluate_model(olive_model, [metric], DEFAULT_CPU_ACCELERATOR)
        else:
            actual_res = docker_system.evaluate_model(olive_model, [metric], DEFAULT_CPU_ACCELERATOR)
            # assert
            self.mock_create_eval_script_mount.called_once_with(container_root_path)
            self.mock_create_model_mount.called_once_with(mock_copy, container_root_path)
            vol_list = [eval_file_mount_str] + model_mount_str_list
            self.mock_create_metric_volumes_list.called_once_with(mock_copy, container_root_path, vol_list)
            self.mock_create_config_file.called_once_with(tempdir, mock_copy, mock_copy, container_root_path)
            self.mock_create_output_mount.called_once_with(tempdir, eval_output_path, container_root_path)
            self.mock_create_evaluate_command.called_once_with(
                eval_file_mount_path, model_mount_path, config_mount_path, output_mount_path, eval_output_name
            )
            self.mock_create_run_command.called_once_with(docker_system.run_params)
            volumes_list.append(config_file_mount_str)
            volumes_list.append(output_mount_str)
            mock_docker_client.containers.run.call_once_with(
                docker_system.image, eval_command, volumes_list, **run_command
            )

            for sub_type in metric.sub_types:
                joint_key = joint_metric_key(metric.name, sub_type.name)
                assert actual_res[joint_key].value == 0.99618
