package com.example.goldentime.api.user;

import com.example.goldentime.user.dto.UserResponseDto;
import com.example.goldentime.user.dto.UserUpdateDto;
import com.example.goldentime.user.dto.VehicleUpdateDto;
import com.example.goldentime.user.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.security.Principal;
import java.util.List;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserApiController {

    private final UserService userService;

    @GetMapping
    public ResponseEntity<List<UserResponseDto>> getAllUsers() {
        return ResponseEntity.ok(userService.findAll());
    }

    @GetMapping("/me")
    public ResponseEntity<UserResponseDto> getCurrentUser(Principal principal) {
        if (principal == null) return ResponseEntity.status(401).build();
        return ResponseEntity.ok(userService.findByLoginId(principal.getName()));
    }

    @PutMapping("/me")
    public ResponseEntity<?> updateProfile(Principal principal, @RequestBody UserUpdateDto updateDto) {
        if (principal == null) return ResponseEntity.status(401).build();
        userService.updateProfile(principal.getName(), updateDto);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/me/vehicles")
    public ResponseEntity<?> addVehicle(Principal principal, @RequestBody VehicleUpdateDto vehicleDto) {
        if (principal == null) return ResponseEntity.status(401).build();
        userService.addVehicle(principal.getName(), vehicleDto);
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/{userId}")
    public ResponseEntity<?> deleteUser(@PathVariable Long userId) {
        userService.deleteUser(userId);
        return ResponseEntity.ok().build();
    }

    @PutMapping("/{userId}/role")
    public ResponseEntity<?> updateRole(@PathVariable Long userId, @RequestParam String role) {
        userService.updateRole(userId, role);
        return ResponseEntity.ok().build();
    }
}
