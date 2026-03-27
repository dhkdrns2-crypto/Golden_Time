package com.example.goldentime.auth;

import com.example.goldentime.user.entity.User;
import com.example.goldentime.user.repository.UserRepository;
import jakarta.annotation.PostConstruct;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

@Component
public class AdminAccountInitializer {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public AdminAccountInitializer(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @PostConstruct
    @Transactional
    public void init() {
        if (!userRepository.existsByLoginId("admin")) {
            User admin = new User();
            admin.setLoginId("admin");
            admin.setPassword(passwordEncoder.encode("1234"));
            admin.setUserName("관리자");
            admin.setRole("ADMIN");
            admin.setPhone("010-0000-0000");
            admin.setEmail("admin@example.com");
            admin.setAddress("시스템");
            userRepository.save(admin);
            System.out.println("Admin account created: admin / 1234");
        } else {
            // 회원가입으로 USER만 들어온 admin 계정 → hasRole("ADMIN") 과 맞추기
            userRepository.findByLoginId("admin").ifPresent(user -> {
                String r = user.getRole() != null ? user.getRole().trim() : "";
                if (!"ADMIN".equalsIgnoreCase(r)) {
                    user.setRole("ADMIN");
                    userRepository.save(user);
                    System.out.println("Admin role restored for user: admin");
                }
            });
        }
    }
}
