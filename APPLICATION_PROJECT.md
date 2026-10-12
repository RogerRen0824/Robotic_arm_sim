# Project description for the application

**Human demonstrations to robot pushing: small-data retargeting and state dynamics**

I recorded 20 phone videos of tabletop object pushing and used a notebook with
known dimensions to extract approximate object-relative trajectories. A direction
guide built exclusively from training videos drives a custom MuJoCo planar
manipulator through a fixed physical-contact controller. I compared direct
pushing, demonstration guidance, goal-weighted guidance and a short PPO residual
policy on five reserved layouts with independently placed target markers.
The weighted guide and PPO both achieved 5/5 successes without object/tool
obstacle contact; PPO did not meaningfully improve the guide. I also trained
action-conditioned state dynamics on simulated interactions informed by the
recorded data. A linear ridge model outperformed the neural predictor, reaching
18.4 mm object-position RMSE over five recursively predicted macro-actions.
The repository includes prepared data, checkpoints, frozen protocols, example
outputs, failures and reproduction instructions. The system uses simulator-state
observations and simplified robot geometry; it does not claim visual policy
learning, whole-arm collision safety, or real-world physics identification.

## Before submitting

- Replace the repository URL with the verified public GitHub URL.
- Review the project description and code so you can explain the design and results.
- Add your own name, current CV, education, availability and other form answers.
- Check the submission date against the deadline extension you received (16 October).
- Verify that the public repository opens while signed out.
- Submit the application only after reviewing the completed form.

## Resource-use note

AI assistance was used for implementation, debugging, experiment execution and
documentation. The phone recordings and goal-marker placement were performed by
the applicant. This note distinguishes data collection from assisted implementation.
