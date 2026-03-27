package com.example.goldentime.config;

import com.example.goldentime.user.entity.User;
import com.example.goldentime.user.repository.UserRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.password.PasswordEncoder;

@Configuration
public class DevBootstrapConfig {

    @Bean
    public CommandLineRunner ensureDefaultAdmin(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        return args -> {
            // 개발 편의: 기본 관리자 계정이 없으면 생성 (loginId: mbc / password: 1234)
            if (!userRepository.existsByLoginId("mbc")) {
                User u = new User();
                u.setLoginId("mbc");
                u.setPassword(passwordEncoder.encode("1234"));
                u.setUserName("관리자");
                u.setPhone("010-0000-0000");
                u.setEmail("admin@example.com");
                u.setAddress("N/A");
                u.setRole("ADMIN");
                userRepository.save(u);
            }
        };
    }
}

