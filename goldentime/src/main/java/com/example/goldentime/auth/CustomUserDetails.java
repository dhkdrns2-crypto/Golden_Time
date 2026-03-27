package com.example.goldentime.auth;

import java.util.Collection;
import java.util.List;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import com.example.goldentime.user.entity.User;

public class CustomUserDetails implements UserDetails {

    private final User user;

    public CustomUserDetails(User user) {
        this.user = user;
    }

    public User getUser() {
        return user;
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        // Spring Security hasRole("ADMIN") → GrantedAuthority "ROLE_ADMIN" 필요
        // DB에 "admin", "ADMIN", "ROLE_admin" 등으로 섞여 있어도 동일하게 매칭되도록 정규화
        String role = user.getRole();
        if (role == null || role.isBlank()) {
            return List.of(new SimpleGrantedAuthority("ROLE_USER"));
        }
        String r = role.trim();
        String suffix = r.startsWith("ROLE_") ? r.substring(5) : r;
        String normalized = "ROLE_" + suffix.toUpperCase();
        return List.of(new SimpleGrantedAuthority(normalized));
    }

    @Override
    public String getPassword() {
        return user.getPassword();
    }

    @Override
    public String getUsername() {
        return user.getLoginId();
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }
}
