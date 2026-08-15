package org.curtinfrc.frc2026;

import choreo.auto.AutoChooser;
import choreo.auto.AutoFactory;
import choreo.auto.AutoRoutine;
import choreo.auto.AutoTrajectory;
import edu.wpi.first.wpilibj.Alert;
import edu.wpi.first.wpilibj.Alert.AlertType;
import edu.wpi.first.wpilibj.Filesystem;
import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.Commands;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.curtinfrc.frc2026.drive.Drive;
import org.curtinfrc.frc2026.drive.Superstructure.Superstructure;
import org.littletonrobotics.junction.Logger;

/** Configures Choreo trajectory following and dashboard autonomous selection. */
public final class Autos {
  private static final String TRAJECTORY_EXTENSION = ".traj";
  private static final String FRIDAY_AUTO = "friday_auto";
  private static final String FRIDAY_2_AUTO = "friday2_auto";
  private static final double AUTO_SHOOT_SPEED_RPM = 2500.0;
  private static final double AUTO_SHOOT_DURATION_SECONDS = 3.0;
  private static final double HUMAN_PLAYER_WAIT_SECONDS = 2.0;

  private final AutoChooser chooser = new AutoChooser("Do Nothing");
  private final Alert noTrajectoriesAlert =
      new Alert(
          "No Choreo trajectories found in deploy/choreo; only Do Nothing is available.",
          AlertType.kWarning);
  private double trajectoryStartTimestampSeconds = 0.0;
  private int eventSequence = 0;

  public Autos(Drive drive, Superstructure shooter) {
    AutoFactory factory =
        new AutoFactory(drive::getPose, drive::setPose, drive::followTrajectory, true, drive);

    configureEventBindings(factory, drive, shooter);

    List<String> trajectoryNames = findDeployedTrajectories();
    noTrajectoriesAlert.set(trajectoryNames.isEmpty());
    for (String trajectoryName : trajectoryNames) {
      System.out.println(trajectoryName);
      chooser.addRoutine(
          trajectoryName, () -> createTrajectoryRoutine(factory, drive, shooter, trajectoryName));
    }
  }

  public AutoChooser getChooser() {
    return chooser;
  }

  /**
   * Global Choreo event-marker names. Add one of these markers to any trajectory to run the
   * corresponding command. Marker names are case-sensitive, so aliases are retained for paths
   * authored with older names.
   */
  private void configureEventBindings(AutoFactory factory, Drive drive, Superstructure shooter) {
    factory
        .bind("Start Intake", loggedEvent("Start Intake", shooter.intake()))
        .bind("Stop Intake", loggedEvent("Stop Intake", shooter.stop()))
        .bind("Finish Intake", loggedEvent("Finish Intake", shooter.stop()))
        .bind(
            "Shoot",
            loggedEvent(
                "Shoot",
                shooter
                    .autoShooter(-AUTO_SHOOT_SPEED_RPM)
                    .withTimeout(AUTO_SHOOT_DURATION_SECONDS)))
        .bind("start intake", loggedEvent("start intake", shooter.intake()))
        .bind("end intake", loggedEvent("end intake", shooter.stop()));
    // .bind("intake", shooter.intake())
    // .bind("stop", shooter.stop())
    // .bind(
    //     "shoot",
    //     shooter.shootBurst(
    //         drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 3))
    // .bind(
    //     "score",
    //     shooter.shootBurst(
    //         drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 1))
    // .bind(
    //     "shoot2",
    //     shooter.shootBurst(
    //         drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 2))
    // .bind(
    //     "shoot3",
    //     shooter.shootBurst(
    //         drive::getDistanceToShotTarget, drive::isReturningFuelToAllianceSide, 3));
  }

  private AutoRoutine createTrajectoryRoutine(
      AutoFactory factory, Drive drive, Superstructure shooter, String trajectoryName) {
    if (trajectoryName.equals(FRIDAY_AUTO)) {
      return createFridayRoutine(factory, drive, shooter);
    }
    if (trajectoryName.equals(FRIDAY_2_AUTO)) {
      return createFriday2Routine(factory, drive, shooter);
    }

    AutoRoutine routine = factory.newRoutine("Choreo: " + trajectoryName);
    AutoTrajectory trajectory = routine.trajectory(trajectoryName);

    // A real AutoRoutine polls Choreo's event loop, allowing bound trajectory markers to fire.
    // Reset immediately before following so the pose is the alliance-aware trajectory start pose.
    routine
        .active()
        .onTrue(
            Commands.sequence(
                Commands.runOnce(() -> logTrajectoryStart(trajectoryName)),
                trajectory.resetOdometry(),
                trajectory.cmd()));
    return routine;
  }

  private AutoRoutine createFridayRoutine(
      AutoFactory factory, Drive drive, Superstructure shooter) {
    AutoRoutine routine = factory.newRoutine("Choreo: " + FRIDAY_AUTO);
    AutoTrajectory[] segments =
        new AutoTrajectory[] {
          routine.trajectory(FRIDAY_AUTO, 0),
          routine.trajectory(FRIDAY_AUTO, 1),
          routine.trajectory(FRIDAY_AUTO, 2),
          routine.trajectory(FRIDAY_AUTO, 3),
          routine.trajectory(FRIDAY_AUTO, 4),
          routine.trajectory(FRIDAY_AUTO, 5)
        };

    routine
        .active()
        .onTrue(
            Commands.sequence(
                Commands.runOnce(() -> logTrajectoryStart(FRIDAY_AUTO)),
                segments[0].resetOdometry(),
                segments[0].cmd()));

    segments[0]
        .done()
        .onTrue(
            stationaryShot(drive, shooter)
                .andThen(segments[1].spawnCmd())
                .withName("Shot 1 Then Continue Path"));
    segments[1]
        .done()
        .onTrue(
            humanPlayerWait(segments[1], drive, shooter)
                .andThen(segments[2].spawnCmd())
                .withName("Human Player Wait Then Continue Path"));
    for (int i = 2; i < segments.length - 1; i++) {
      int shotNumber = i;
      AutoTrajectory nextSegment = segments[i + 1];
      segments[i]
          .done()
          .onTrue(
              stationaryShot(drive, shooter)
                  .andThen(nextSegment.spawnCmd())
                  .withName("Shot " + shotNumber + " Then Continue Path"));
    }
    return routine;
  }

  private AutoRoutine createFriday2Routine(
      AutoFactory factory, Drive drive, Superstructure shooter) {
    AutoRoutine routine = factory.newRoutine("Choreo: " + FRIDAY_2_AUTO);
    AutoTrajectory driveToFirstShot = routine.trajectory(FRIDAY_2_AUTO, 0);
    AutoTrajectory driveToHumanPlayer = routine.trajectory(FRIDAY_2_AUTO, 1);
    AutoTrajectory driveToSecondShot = routine.trajectory(FRIDAY_2_AUTO, 2);
    AutoTrajectory driveAfterSecondShot = routine.trajectory(FRIDAY_2_AUTO, 3);

    routine
        .active()
        .onTrue(
            Commands.sequence(
                Commands.runOnce(() -> logTrajectoryStart(FRIDAY_2_AUTO)),
                driveToFirstShot.resetOdometry(),
                driveToFirstShot.cmd()));

    // Each shooting segment ends at a Choreo stop point. Explicitly stop both motors before
    // scheduling any movement so the shooter remains off for the whole path between shots.
    driveToFirstShot
        .done()
        .onTrue(
            stationaryShot(drive, shooter)
                .andThen(driveToHumanPlayer.spawnCmd())
                .withName("First Shot Then Continue Path"));

    driveToHumanPlayer
        .done()
        .onTrue(
            humanPlayerWait(driveToHumanPlayer, drive, shooter)
                .andThen(driveToSecondShot.spawnCmd())
                .withName("Human Player Wait Then Continue Path"));

    driveToSecondShot
        .done()
        .onTrue(
            stationaryShot(drive, shooter)
                .andThen(driveAfterSecondShot.spawnCmd())
                .withName("Second Shot Then Continue Path"));

    return routine;
  }

  private Command stationaryShot(Drive drive, Superstructure shooter) {
    return Commands.waitUntil(drive::isStopped)
        .andThen(
            loggedEvent(
                "Shoot",
                shooter.autoShooter(AUTO_SHOOT_SPEED_RPM).withTimeout(AUTO_SHOOT_DURATION_SECONDS)))
        .andThen(shooter.stopOnce());
  }

  private Command humanPlayerWait(
      AutoTrajectory arrivalSegment, Drive drive, Superstructure shooter) {
    return drive
        .settleToPose(arrivalSegment::getFinalPose)
        .andThen(Commands.waitUntil(drive::isStopped))
        .andThen(
            loggedEvent(
                "Human Player Wait",
                Commands.sequence(
                    shooter.stopOnce(), Commands.waitSeconds(HUMAN_PLAYER_WAIT_SECONDS))));
  }

  private Command loggedEvent(String markerName, Command command) {
    return Commands.sequence(
            Commands.runOnce(
                () -> {
                  eventSequence++;
                  Logger.recordOutput("Autonomous/Event/Sequence", eventSequence);
                  Logger.recordOutput("Autonomous/Event/Name", markerName);
                  Logger.recordOutput(
                      "Autonomous/Event/ElapsedSeconds",
                      Timer.getFPGATimestamp() - trajectoryStartTimestampSeconds);
                }),
            command)
        .withName("Auto Event: " + markerName);
  }

  private void logTrajectoryStart(String trajectoryName) {
    trajectoryStartTimestampSeconds = Timer.getFPGATimestamp();
    eventSequence = 0;
    Logger.recordOutput("Autonomous/Trajectory/Name", trajectoryName);
    Logger.recordOutput(
        "Autonomous/Trajectory/StartTimestampSeconds", trajectoryStartTimestampSeconds);
    Logger.recordOutput("Autonomous/Event/Sequence", eventSequence);
    Logger.recordOutput("Autonomous/Event/Name", "");
    Logger.recordOutput("Autonomous/Event/ElapsedSeconds", 0.0);
  }

  private static List<String> findDeployedTrajectories() {
    Path choreoDirectory = Filesystem.getDeployDirectory().toPath().resolve("choreo");
    if (!Files.isDirectory(choreoDirectory)) {
      System.out.println(
          "========================================================================================");
      return List.of();
    }

    try (var files = Files.list(choreoDirectory)) {
      System.out.println("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");

      return files
          .filter(Files::isRegularFile)
          .map(path -> path.getFileName().toString())
          .filter(name -> name.endsWith(TRAJECTORY_EXTENSION))
          .map(name -> name.substring(0, name.length() - TRAJECTORY_EXTENSION.length()))
          .sorted()
          .toList();
    } catch (IOException exception) {
      return List.of();
    }
  }
}
