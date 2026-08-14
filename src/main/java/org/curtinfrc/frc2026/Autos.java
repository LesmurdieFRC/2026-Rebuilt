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
  private static final String FRIDAY_2_AUTO = "friday2_auto";
  private static final double AUTO_SHOOT_SPEED_RPM = 2500.0;
  private static final double AUTO_SHOOT_DURATION_SECONDS = 2.0;

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
          trajectoryName, () -> createTrajectoryRoutine(factory, shooter, trajectoryName));
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
                shooter.shooter(AUTO_SHOOT_SPEED_RPM).withTimeout(AUTO_SHOOT_DURATION_SECONDS)))
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
      AutoFactory factory, Superstructure shooter, String trajectoryName) {
    if (trajectoryName.equals(FRIDAY_2_AUTO)) {
      return createFriday2Routine(factory, shooter);
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

  private AutoRoutine createFriday2Routine(AutoFactory factory, Superstructure shooter) {
    AutoRoutine routine = factory.newRoutine("Choreo: " + FRIDAY_2_AUTO);
    AutoTrajectory driveToFirstShot = routine.trajectory(FRIDAY_2_AUTO, 0);
    AutoTrajectory driveToSecondShot = routine.trajectory(FRIDAY_2_AUTO, 1);
    AutoTrajectory driveAfterSecondShot = routine.trajectory(FRIDAY_2_AUTO, 2);

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
            loggedEvent(
                    "Shoot",
                    shooter.shooter(AUTO_SHOOT_SPEED_RPM).withTimeout(AUTO_SHOOT_DURATION_SECONDS))
                .andThen(shooter.stopOnce())
                .andThen(driveToSecondShot.spawnCmd())
                .withName("First Shot Then Continue Path"));

    driveToSecondShot
        .done()
        .onTrue(
            loggedEvent(
                    "Shoot",
                    shooter.shooter(AUTO_SHOOT_SPEED_RPM).withTimeout(AUTO_SHOOT_DURATION_SECONDS))
                .andThen(shooter.stopOnce())
                .andThen(driveAfterSecondShot.spawnCmd())
                .withName("Second Shot Then Continue Path"));

    return routine;
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
