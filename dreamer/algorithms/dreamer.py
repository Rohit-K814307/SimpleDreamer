import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dreamer.modules.model import RSSM# , RewardModel, ContinueModel
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
# from dreamer.modules.actor import Actor
# from dreamer.modules.critic import Critic

from dreamer.utils.utils import (
#     compute_lambda_values,
     create_normal_dist,
#     DynamicInfos,
)
from dreamer.utils.buffer import ReplayBuffer


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        logger,
        device,
        config,
        train_buffer=None,
        val_buffer=None,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)

        # self.reward_predictor = RewardModel(config).to(self.device)
        # if config.parameters.dreamer.use_continue_flag:
        #     self.continue_predictor = ContinueModel(config).to(self.device)
        # self.actor = Actor(discrete_action_bool, action_size, config).to(self.device)
        # self.critic = Critic(config).to(self.device)

        if train_buffer is not None:
            self.buffer = train_buffer
            self.val_buffer = val_buffer
        else:
            self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)
            self.val_buffer = None

        self.config = config.parameters.dreamer

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            #+ list(self.reward_predictor.parameters())
        )
        # if self.config.use_continue_flag:
        #     self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        # self.actor_optimizer = torch.optim.Adam(
        #     self.actor.parameters(), lr=self.config.actor_learning_rate
        # )
        # self.critic_optimizer = torch.optim.Adam(
        #     self.critic.parameters(), lr=self.config.critic_learning_rate
        # )

        # self.continue_criterion = nn.BCELoss()

        # self.dynamic_learning_infos = DynamicInfos(self.device)
        # self.behavior_learning_infos = DynamicInfos(self.device)

        self.logger = logger
        self.num_total_episode = 0

    def _set_train_mode(self):
        self.encoder.train()
        self.decoder.train()
        self.rssm.train()

    def _set_eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
        self.rssm.eval()

    def save_checkpoint(self, checkpoint_dir, epoch, total_steps):
        """Save model checkpoint (encoder, decoder, rssm)."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "total_steps": total_steps,
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "rssm": self.rssm.state_dict(),
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        latest_path = checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"  Saved checkpoint: {checkpoint_path.name}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.rssm.load_state_dict(checkpoint["rssm"])
        
        return checkpoint.get("epoch", 0), checkpoint.get("total_steps", 0)

    def train(self, env=None, viz_interval=50):
        """Legacy iteration-based training (for backwards compatibility)."""
        total_steps = 0
        for iteration in range(self.config.train_iterations):
            self._set_train_mode()
            
            # Accumulate losses over collect_interval
            accumulated_losses = {"reconstruction_loss": 0.0, "kl_loss": 0.0, "model_loss": 0.0}
            
            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                losses = self.dynamic_learning(data)
                for k in accumulated_losses:
                    accumulated_losses[k] += losses[k]
                total_steps += 1
            
            # Average over collect_interval
            num_steps = self.config.collect_interval
            avg_losses = {k: v / num_steps for k, v in accumulated_losses.items()}
            
            print(f"Iter {iteration}/{self.config.train_iterations}, "
                  f"recon={avg_losses['reconstruction_loss']:.4f}, kl={avg_losses['kl_loss']:.4f}", flush=True)
            
            # Validation and visualizations at viz_interval
            if iteration % viz_interval == 0:
                self._set_eval_mode()
                
                if self.logger is not None:
                    self.logger.log_scalar("loss/reconstruction", avg_losses["reconstruction_loss"], total_steps)
                    self.logger.log_scalar("loss/kl", avg_losses["kl_loss"], total_steps)
                    self.logger.log_scalar("loss/total", avg_losses["model_loss"], total_steps)
                
                if self.val_buffer is not None:
                    val_losses = self._compute_validation_loss()
                    if self.logger is not None:
                        self.logger.log_scalar("val_loss/reconstruction", val_losses["reconstruction_loss"], total_steps)
                        self.logger.log_scalar("val_loss/kl", val_losses["kl_loss"], total_steps)
                        self.logger.log_scalar("val_loss/total", val_losses["model_loss"], total_steps)
                    print(f"  [Val] recon={val_losses['reconstruction_loss']:.4f}, kl={val_losses['kl_loss']:.4f}", flush=True)
                
                # Visualizations for both train and val
                if self.logger is not None:
                    self._log_visualizations(data, iteration, prefix="train_")
                    if self.val_buffer is not None:
                        val_data = self.val_buffer.sample(
                            self.config.batch_size, self.config.batch_length
                        )
                        self._log_visualizations(val_data, iteration, prefix="val_")
                    self.logger.save_loss_curves()
                    self.save_checkpoint(self.logger.log_dir / "checkpoints", iteration, total_steps)

    def train_epochs(self, train_loader, val_loader=None, num_epochs=100,
                     viz_interval=1):
        """
        Epoch-based training using DataLoaders.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train
            viz_interval: Epochs between validation and visualizations (both train and val)
        """
        total_steps = 0
        
        for epoch in range(num_epochs):
            self._set_train_mode()
            
            # Training epoch
            epoch_losses = {"reconstruction_loss": 0.0, "kl_loss": 0.0, "model_loss": 0.0}
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
            for batch_idx, data in enumerate(pbar):
                losses = self.dynamic_learning(data)
                for k in epoch_losses:
                    epoch_losses[k] += losses[k]
                num_batches += 1
                total_steps += 1
                pbar.set_postfix(recon=f"{losses['reconstruction_loss']:.4f}", kl=f"{losses['kl_loss']:.4f}")
            
            # Average over epoch
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            
            print(f"Epoch {epoch + 1}/{num_epochs} ({num_batches} batches), "
                  f"recon={avg_losses['reconstruction_loss']:.4f}, kl={avg_losses['kl_loss']:.4f}", flush=True)
            
            # Log training losses every epoch
            if self.logger is not None:
                self.logger.log_scalar("loss/reconstruction", avg_losses["reconstruction_loss"], total_steps)
                self.logger.log_scalar("loss/kl", avg_losses["kl_loss"], total_steps)
                self.logger.log_scalar("loss/total", avg_losses["model_loss"], total_steps)
            
            # Validation and visualizations (both train and val) at viz_interval
            if (epoch + 1) % viz_interval == 0:
                self._set_eval_mode()
                
                # Validation loss
                if val_loader is not None:
                    val_losses = self._compute_validation_loss_epoch(val_loader)
                    
                    if self.logger is not None:
                        self.logger.log_scalar("val_loss/reconstruction", val_losses["reconstruction_loss"], total_steps)
                        self.logger.log_scalar("val_loss/kl", val_losses["kl_loss"], total_steps)
                        self.logger.log_scalar("val_loss/total", val_losses["model_loss"], total_steps)
                    
                    print(f"  [Val] recon={val_losses['reconstruction_loss']:.4f}, "
                          f"kl={val_losses['kl_loss']:.4f}", flush=True)
                
                # Visualizations for both train and val
                if self.logger is not None:
                    self._log_visualizations(data, epoch, prefix="train_")
                    if val_loader is not None:
                        val_data = next(iter(val_loader))
                        self._log_visualizations(val_data, epoch, prefix="val_")
                    self.logger.save_loss_curves()
                    self.save_checkpoint(self.logger.log_dir / "checkpoints", epoch, total_steps)

    # def evaluate(self, env):
    #     self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(data.action.shape[0])

        # add the [current yaw, current robot velocity in x, current robot velocity in y] state observations
        embedded_observation = self.encoder(data.observation, data.state)

        priors = []
        posteriors = []
        deterministics = []
        prior_means = []
        prior_stds = []
        posterior_means = []
        posterior_stds = []

        for t in range(1, data.observation.shape[1]):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                embedded_observation[:, t], deterministic
            )

            # self.dynamic_learning_infos.append(
            #     priors=prior,
            #     prior_dist_means=prior_dist.mean,
            #     prior_dist_stds=prior_dist.scale,
            #     posteriors=posterior,
            #     posterior_dist_means=posterior_dist.mean,
            #     posterior_dist_stds=posterior_dist.scale,
            #     deterministics=deterministic,
            # )

            priors.append(prior)
            posteriors.append(posterior)
            deterministics.append(deterministic)
            prior_means.append(prior_dist.mean)
            prior_stds.append(prior_dist.scale)
            posterior_means.append(posterior_dist.mean)
            posterior_stds.append(posterior_dist.scale)

            prior = posterior.detach()

        posterior_info = {
            "priors": torch.stack(priors, dim=1),
            "posteriors": torch.stack(posteriors, dim=1),
            "deterministics": torch.stack(deterministics, dim=1),
            "prior_means": torch.stack(prior_means, dim=1),
            "prior_stds": torch.stack(prior_stds, dim=1),
            "posterior_means": torch.stack(posterior_means, dim=1),
            "posterior_stds": torch.stack(posterior_stds, dim=1),
        }
        
        losses = self._model_update(data, posterior_info)

        return losses

    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            posterior_info["posteriors"], posterior_info["deterministics"]
        )
        # reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
        #     data.observation[:, 1:]
        # )
        # if self.config.use_continue_flag:
        #     continue_dist = self.continue_predictor(
        #         posterior_info.posteriors, posterior_info.deterministics
        #     )
        #     continue_loss = self.continue_criterion(
        #         continue_dist.probs, 1 - data.done[:, 1:]
        #     )

        # reward_dist = self.reward_predictor(
        #     posterior_info.posteriors, posterior_info.deterministics
        # )
        # reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info["prior_means"],
            posterior_info["prior_stds"],
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info["posterior_means"],
            posterior_info["posterior_stds"],
            event_shape=1,
        )

        reconstruction_loss = -reconstructed_observation_dist.log_prob(
            data.observation[:, 1:].view(reconstructed_observation_dist.mean.shape)
        ).mean()

        kl_loss = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        kl_loss = torch.clamp(kl_loss, min=self.config.free_nats)
        kl_loss = kl_loss.sum(dim=-1).mean()

        model_loss = self.config.kl_divergence_scale * kl_loss + reconstruction_loss

        # import pdb; pdb.set_trace()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step() 

        return {
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_loss": kl_loss.item(),
            "model_loss": model_loss.item(),
        }

    @torch.no_grad()
    def _compute_validation_loss(self, num_batches: int = 5):
        """Compute validation loss over multiple batches without gradients (legacy sample-based)."""
        total_recon = 0.0
        total_kl = 0.0
        total_model = 0.0
        
        for _ in range(num_batches):
            data = self.val_buffer.sample(
                self.config.batch_size, self.config.batch_length
            )
            
            recon, kl, model = self._compute_losses_for_batch(data)
            total_recon += recon
            total_kl += kl
            total_model += model
        
        return {
            "reconstruction_loss": total_recon / num_batches,
            "kl_loss": total_kl / num_batches,
            "model_loss": total_model / num_batches,
        }
    
    @torch.no_grad()
    def _compute_validation_loss_epoch(self, val_loader, max_batches: int = None):
        """Compute validation loss over DataLoader without gradients."""
        total_recon = 0.0
        total_kl = 0.0
        total_model = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            recon, kl, model = self._compute_losses_for_batch(data)
            total_recon += recon
            total_kl += kl
            total_model += model
            num_batches += 1
        
        return {
            "reconstruction_loss": total_recon / num_batches,
            "kl_loss": total_kl / num_batches,
            "model_loss": total_model / num_batches,
        }
    
    @torch.no_grad()
    def _compute_losses_for_batch(self, data):
        """Compute losses for a single batch (shared by validation methods)."""
        prior, deterministic = self.rssm.recurrent_model_input_init(data.action.shape[0])
        embedded_observation = self.encoder(data.observation, data.state)
        
        prior_means = []
        prior_stds = []
        posterior_means = []
        posterior_stds = []
        posteriors = []
        deterministics = []
        
        for t in range(1, data.observation.shape[1]):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                embedded_observation[:, t], deterministic
            )
            
            posteriors.append(posterior)
            deterministics.append(deterministic)
            prior_means.append(prior_dist.mean)
            prior_stds.append(prior_dist.scale)
            posterior_means.append(posterior_dist.mean)
            posterior_stds.append(posterior_dist.scale)
            
            prior = posterior
        
        posteriors = torch.stack(posteriors, dim=1)
        deterministics = torch.stack(deterministics, dim=1)
        
        reconstructed_observation_dist = self.decoder(posteriors, deterministics)
        
        prior_dist = create_normal_dist(
            torch.stack(prior_means, dim=1),
            torch.stack(prior_stds, dim=1),
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            torch.stack(posterior_means, dim=1),
            torch.stack(posterior_stds, dim=1),
            event_shape=1,
        )
        
        reconstruction_loss = -reconstructed_observation_dist.log_prob(
            data.observation[:, 1:].view(reconstructed_observation_dist.mean.shape)
        ).mean()
        
        kl_loss = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        kl_loss = torch.clamp(kl_loss, min=self.config.free_nats)
        kl_loss = kl_loss.sum(dim=-1).mean()
        
        model_loss = self.config.kl_divergence_scale * kl_loss + reconstruction_loss
        
        return reconstruction_loss.item(), kl_loss.item(), model_loss.item()

    @torch.no_grad()
    def predict_next_state(self, observation, state, action, prev_state=None):
        """
        observation: raw observation (numpy or tensor)
        action: action tensor (already chosen by you)
        prev_state: (posterior, deterministic) or None
        """

        if prev_state is None:
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
        else:
            posterior = prev_state[0].to(self.device)
            deterministic = prev_state[1].to(self.device)

        observation = torch.as_tensor(observation, device=self.device).float()
        obs_emb = self.encoder(observation.unsqueeze(0), state)
        action = action.unsqueeze(0) if action.dim() == 1 else action
        self.rssm.recurrent_model(posterior, action, deterministic)
        _, posterior = self.rssm.representation_model(obs_emb, deterministic)

        recon_obs_dist = self.decoder(posterior, deterministic)

        return {
            "posterior": posterior,
            "deterministic": deterministic,
            "reconstructed_obs_mean": recon_obs_dist.mean,
            "reconstructed_obs_dist": recon_obs_dist,
        }

    @torch.no_grad()
    def imagine_horizon(self, initial_observation, initial_state, action_sequence):
        """
        Imagine n steps into the future using only the prior (no real observations).
        
        Args:
            initial_observation: (C, H, W) or (1, C, H, W) tensor - the starting observation
            action_sequence: (n, action_size) or (1, n, action_size) tensor - actions to take
        
        Returns:
            dict with:
                - imagined_observations: (n, C, H, W) tensor of decoded predictions
                - priors: (n, stochastic_size) latent states
                - deterministics: (n, deterministic_size) hidden states
        """
        # Ensure correct shapes
        initial_observation = torch.as_tensor(initial_observation, device=self.device).float()
        initial_state = torch.as_tensor(initial_state, device = self.device).float()
        if initial_observation.dim() == 3:
            initial_observation = initial_observation.unsqueeze(0)  # (1, C, H, W)
        
        action_sequence = torch.as_tensor(action_sequence, device=self.device).float()
        if action_sequence.dim() == 2:
            action_sequence = action_sequence.unsqueeze(0)  # (1, n, action_size)
        
        batch_size = initial_observation.shape[0]
        horizon = action_sequence.shape[1]
        
        # Initialize hidden states
        stochastic, deterministic = self.rssm.recurrent_model_input_init(batch_size)
        
        # Encode initial observation to get starting posterior
        initial_state = initial_state.unsqueeze(1)
        initial_observation = initial_observation.unsqueeze(1)
        embedded_obs = self.encoder(initial_observation, initial_state)
        # Get initial posterior from real observation
        _, stochastic = self.rssm.representation_model(embedded_obs.squeeze(1), deterministic)
        
        # Roll forward using PRIOR path (no real observations)
        imagined_observations = []
        priors = []
        deterministics = []
        
        for t in range(horizon):
            # Step 1: Update deterministic state using recurrent model
            deterministic = self.rssm.recurrent_model(stochastic, action_sequence[:, t], deterministic)
            
            # Step 2: Predict next stochastic state using TRANSITION model (prior)
            # NOT representation model (that would need real observation)
            _, stochastic = self.rssm.transition_model(deterministic)
            
            # Step 3: Decode the imagined state to get predicted observation
            decoded_dist = self.decoder(stochastic, deterministic)
            
            # Store results
            imagined_observations.append(decoded_dist.mean)
            priors.append(stochastic)
            deterministics.append(deterministic)
        
        return {
            "imagined_observations": torch.stack(imagined_observations, dim=1).squeeze(0),  # (n, C, H, W)
            "priors": torch.stack(priors, dim=1).squeeze(0),  # (n, stochastic_size)
            "deterministics": torch.stack(deterministics, dim=1).squeeze(0),  # (n, deterministic_size)
        }

    def _unnormalize_img(self, img, img_mean, img_std):
        """Un-normalize image tensor: img * std + mean, then scale to [0, 1].
        
        Args:
            img: tensor of any shape with normalized pixel values
            img_mean: broadcastable mean tensor (computed on [0, 255] range)
            img_std: broadcastable std tensor (computed on [0, 255] range)
        
        Returns:
            Tensor in [0, 1] range suitable for display.
        """
        # Unnormalize back to [0, 255] range, then scale to [0, 1]
        img_255 = img * img_std + img_mean
        return (img_255 / 255.0).clamp(0, 1)

    @torch.no_grad()
    def _log_visualizations(self, data, iteration, num_samples=1, prefix=""):
        """Log reconstruction and imagination visualizations for all timesteps."""
        split = "train" if prefix == "train_" else "val"
        
        obs = data.observation[:num_samples]
        actions = data.action[:num_samples]
        states = data.state[:num_samples]
        img_mean = data.img_mean[:num_samples]
        img_std = data.img_std[:num_samples]
        
        # Reconstruction: all timesteps
        recon_result = self._visualize_reconstruction(obs, states, actions, img_mean, img_std)
        if recon_result is not None:
            gt_frames = recon_result["ground_truth"]  # (T, C, H, W)
            pred_frames = recon_result["reconstructed"]  # (T, C, H, W)
            self.logger.log_image_sequence(gt_frames, iteration, split, "recon", "gt")
            self.logger.log_image_sequence(pred_frames, iteration, split, "recon", "pred")
            self.logger.log_comparison_gif(gt_frames, pred_frames, iteration, split, "recon")
        
        # Imagination: all timesteps
        imag_result = self._visualize_imagination(obs, states, actions, img_mean, img_std)
        if imag_result is not None:
            gt_frames = imag_result["ground_truth"]  # (horizon, C, H, W)
            pred_frames = imag_result["predicted"]  # (horizon, C, H, W)
            self.logger.log_image_sequence(gt_frames, iteration, split, "imag", "gt")
            self.logger.log_image_sequence(pred_frames, iteration, split, "imag", "pred")
            self.logger.log_comparison_gif(gt_frames, pred_frames, iteration, split, "imag")

    @torch.no_grad()
    def _visualize_reconstruction(self, observations, states, actions, img_mean, img_std):
        """
        Reconstruct all frames using posterior (with access to observations).
        
        Returns T frames: t00 = input, t01 onwards = reconstructed.
        """
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        
        if seq_len < 2:
            return None
        
        prior, deterministic = self.rssm.recurrent_model_input_init(batch_size)
        embedded_obs = self.encoder(observations, states)
        
        posteriors = []
        deterministics = []
        
        for t in range(1, seq_len):
            deterministic = self.rssm.recurrent_model(prior, actions[:, t - 1], deterministic)
            _, prior = self.rssm.transition_model(deterministic)
            _, posterior = self.rssm.representation_model(embedded_obs[:, t], deterministic)
            posteriors.append(posterior)
            deterministics.append(deterministic)
            prior = posterior
        
        posteriors = torch.stack(posteriors, dim=1)
        deterministics = torch.stack(deterministics, dim=1)
        
        reconstructed_dist = self.decoder(posteriors, deterministics)
        reconstructed = reconstructed_dist.mean  # (batch, T-1, C, H, W)
        
        # Unnormalize - take first sample from batch
        mean = img_mean[0].squeeze(0)  # (1, 1, 1)
        std = img_std[0].squeeze(0)
        
        # Input observation (t=0)
        input_obs = self._unnormalize_img(observations[0, 0:1], mean, std)  # (1, C, H, W)
        
        # Ground truth: input + observations[1:] (T frames total)
        gt_rest = self._unnormalize_img(observations[0, 1:], mean, std)  # (T-1, C, H, W)
        ground_truth = torch.cat([input_obs, gt_rest], dim=0)  # (T, C, H, W)
        
        # Reconstructed: input + reconstructed (T frames total)
        recon_rest = self._unnormalize_img(reconstructed[0], mean, std)  # (T-1, C, H, W)
        recon = torch.cat([input_obs, recon_rest], dim=0)  # (T, C, H, W)
        
        return {"ground_truth": ground_truth, "reconstructed": recon}

    @torch.no_grad()
    def _visualize_imagination(self, observations, states, actions, img_mean, img_std):
        """
        Imagine forward from first observation using only actions (no observations).
        
        Returns T frames: t00 = input, t01 onwards = imagined.
        """
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        
        horizon = seq_len - 1  # Use full sequence length
        if horizon < 1:
            return None
        
        initial_obs = observations[:, 0]
        initial_state = states[:, 0]
        action_seq = actions[:, :horizon]
        
        result = self.imagine_horizon(initial_obs, initial_state, action_seq)
        predicted = result["imagined_observations"]  # (horizon, C, H, W) or (batch, horizon, C, H, W)
        
        if predicted.dim() == 4:
            predicted = predicted.unsqueeze(0)
        
        # Unnormalize - take first sample from batch
        mean = img_mean[0].squeeze(0)  # (1, 1, 1)
        std = img_std[0].squeeze(0)
        
        # Input observation (t=0)
        input_obs = self._unnormalize_img(observations[0, 0:1], mean, std)  # (1, C, H, W)
        
        # Ground truth: input + observations[1:horizon+1] (T frames total)
        gt_rest = self._unnormalize_img(observations[0, 1:horizon + 1], mean, std)  # (horizon, C, H, W)
        ground_truth = torch.cat([input_obs, gt_rest], dim=0)  # (T, C, H, W)
        
        # Predicted: input + imagined (T frames total)
        pred_rest = self._unnormalize_img(predicted[0], mean, std)  # (horizon, C, H, W)
        predicted_full = torch.cat([input_obs, pred_rest], dim=0)  # (T, C, H, W)
        
        return {"ground_truth": ground_truth, "predicted": predicted_full}

    # def behavior_learning(self, states, deterministics):
    #     """
    #     #TODO : last posterior truncation(last can be last step)
    #     posterior shape : (batch, timestep, stochastic)
    #     """
    #     state = states.reshape(-1, self.config.stochastic_size)
    #     deterministic = deterministics.reshape(-1, self.config.deterministic_size)

    #     # continue_predictor reinit
    #     for t in range(self.config.horizon_length):
    #         action = self.actor(state, deterministic)
    #         deterministic = self.rssm.recurrent_model(state, action, deterministic)
    #         _, state = self.rssm.transition_model(deterministic)
    #         self.behavior_learning_infos.append(
    #             priors=state, deterministics=deterministic
    #         )

    #     self._agent_update(self.behavior_learning_infos.get_stacked())

    # def _agent_update(self, behavior_learning_infos):
    #     predicted_rewards = self.reward_predictor(
    #         behavior_learning_infos.priors, behavior_learning_infos.deterministics
    #     ).mean
    #     values = self.critic(
    #         behavior_learning_infos.priors, behavior_learning_infos.deterministics
    #     ).mean

    #     if self.config.use_continue_flag:
    #         continues = self.continue_predictor(
    #             behavior_learning_infos.priors, behavior_learning_infos.deterministics
    #         ).mean
    #     else:
    #         continues = self.config.discount * torch.ones_like(values)

    #     lambda_values = compute_lambda_values(
    #         predicted_rewards,
    #         values,
    #         continues,
    #         self.config.horizon_length,yeah
    #     nn.utils.clip_grad_norm_(
    #         self.actor.parameters(),
    #         self.config.clip_grad,
    #         norm_type=self.config.grad_norm_type,
    #     )
    #     self.actor_optimizer.step()
    #     value_dist = self.critic(
    #         behavior_learning_infos.priors.detach()[:, :-1],
    #         behavior_learning_infos.deterministics.detach()[:, :-1],
    #     )
    #     value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

    #     self.critic_optimizer.zero_grad()
    #     value_loss.backward()
    #     nn.utils.clip_grad_norm_(
    #         self.critic.parameters(),
    #         self.config.clip_grad,
    #         norm_type=self.config.grad_norm_type,
    #     )
    #     self.critic_optimizer.step()

    # @torch.no_grad()
    # def environment_interaction(self, env, num_interaction_episodes, train=True):
    #     for epi in range(num_interaction_episodes):
    #         posterior, deterministic = self.rssm.recurrent_model_input_init(1)
    #         action = torch.zeros(1, self.action_size).to(self.device)

    #         observation = env.reset()
    #         embedded_observation = self.encoder(
    #             torch.from_numpy(observation).float().to(self.device)
    #         )

    #         score = 0
    #         score_lst = np.array([])
    #         done = False

    #         while not done:
    #             deterministic = self.rssm.recurrent_model(
    #                 posterior, action, deterministic
    #             )
    #             embedded_observation = embedded_observation.reshape(1, -1)
    #             _, posterior = self.rssm.representation_model(
    #                 embedded_observation, deterministic
    #             )
    #             action = self.actor(posterior, deterministic).detach()

    #             if self.discrete_action_bool:
    #                 buffer_action = action.cpu().numpy()
    #                 env_action = buffer_action.argmax()

    #             else:
    #                 buffer_action = action.cpu().numpy()[0]
    #                 env_action = buffer_action

    #             next_observation, reward, done, info = env.step(env_action)
    #             if train:
    #                 self.buffer.add(
    #                     observation, buffer_action, reward, next_observation, done
    #                 )
    #             score += reward
    #             embedded_observation = self.encoder(
    #                 torch.from_numpy(next_observation).float().to(self.device)
    #             )
    #             observation = next_observation
    #             if done:
    #                 if train:
    #                     self.num_total_episode += 1
    #                     self.writer.add_scalar(
    #                         "training score", score, self.num_total_episode
    #                     )
    #                 else:
    #                     score_lst = np.append(score_lst, score)
    #                 break
    #     if not train:
    #         evaluate_score = score_lst.mean()
    #         print("evaluate score : ", evaluate_score)
    #         self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)
