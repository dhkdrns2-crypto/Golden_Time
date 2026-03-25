package com.example.goldentime.user.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
public class UserUpdateDto {
    private String userName;
    private String phone;
    private String email;
    private String address;
    private String password; // 선택적 수정
    private List<VehicleUpdateDto> vehicles;

    public String getUserName() { return userName; }
    public void setUserName(String userName) { this.userName = userName; }
    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }
    public List<VehicleUpdateDto> getVehicles() { return vehicles; }
    public void setVehicles(List<VehicleUpdateDto> vehicles) { this.vehicles = vehicles; }
}
